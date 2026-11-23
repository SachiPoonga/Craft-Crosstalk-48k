"""CrossStemDebleedRefiner (v2): Full-Bin 48 kHz 5-Reference Residue & Bleed Canceller.

Chains directly after frozen HarmonicSlot-TF v3 (Craft Crosstalk-48k) to scrub
inter-speaker ghost bleed, pause residue, 2-10 kHz sibilant leakage, and STFT
metallic grain at full 1,025-bin STFT resolution while strictly preserving input
speaker slot ordering.

Key design principles:
  1. Full 1,025-bin (48 kHz, n_fft=2048, hop=256) stereo complex STFT processing.
  2. Independent 5-reference cancellation with deterministic energy sorting:
       - For each Voice v: [Y_self, Y_o1, Y_o2, Y_music, Y_amb, Y_mix]
         where (Y_o1, Y_o2) are the other 2 voices sorted by descending RMS so the
         refiner is 100% invariant to which voice slots are active/empty, and can
         cancel Speaker 2 & Speaker 3 at independent amplitudes & phases!
       - For Music / Ambience: [Y_self, Y_v1, Y_v2, Y_v3, Y_other_bg, Y_mix]
         with (Y_v1 >= Y_v2 >= Y_v3) sorted by descending RMS so each voice's
         bleed into Music & Ambience is cancelled independently.
  3. Physics features: log-compressed Re/Im + per-reference magnitudes + target
     dominance ratio + exact per-reference complex phase coherence.
  4. 2D Frequency-only U-Net (keeps full Time resolution T) + fp32-protected
     Dual-Path Freq/Time BiGRU bottleneck (no cuDNN fp16 underflow under bf16).
  5. Identity-initialized bounded complex multi-reference cancellation head +
     tiny RMS-bounded 1D waveform de-grain polish (self-cancellation stripped).
  6. Strict slot-order & silent-slot preservation: empty voice slots stay 0.0.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _apply_bounded_wave_residual(
    base: torch.Tensor,
    raw_delta: torch.Tensor,
    max_ratio: float = 0.15,
    rms_win: int = 257,
) -> torch.Tensor:
    """Apply a local-RMS-bounded waveform residual with negative self-projection stripped.

    Gating by local frame RMS (~5.3 ms at 48 kHz) instead of whole-clip RMS guarantees the
    1D waveform polish can never re-inject bleed or noise into silent pauses scrubbed by STFT.
    """
    lead = base.shape[:-1]
    N = base.shape[-1]
    b_flat = base.reshape(-1, 1, N).float()
    pad = rms_win // 2
    local_rms = (
        F.avg_pool1d(b_flat.pow(2), kernel_size=rms_win, stride=1, padding=pad)
        .add(1e-12)
        .sqrt()
        .reshape(*lead, N)
    )
    delta = max_ratio * local_rms * torch.tanh(raw_delta.float())
    proj = (delta * base).sum(dim=(-2, -1), keepdim=True) / (
        base.pow(2).sum(dim=(-2, -1), keepdim=True) + 1e-8
    )
    return base + delta - F.relu(-proj) * base


class _ResBlock2D(nn.Module):
    """GroupNorm + SiLU 2D Conv residual block preserving (F, T) spatial dimensions."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        g = math.gcd(channels, groups)
        self.net = nn.Sequential(
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _DualPathBottleneck(nn.Module):
    """Frequency-axis BiGRU (harmonic stacks) + Time-axis BiGRU (pause context) in fp32."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm_f = nn.LayerNorm(channels)
        self.gru_f = nn.GRU(channels, channels // 2, batch_first=True, bidirectional=True)
        self.norm_t = nn.LayerNorm(channels)
        self.gru_t = nn.GRU(channels, channels // 2, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run bottleneck BiGRUs in fp32 so cuDNN's internal fp16 RNN autocast never
        # underflows small gradients when training in bf16 without GradScaler.
        B, C, F_b, T = x.shape
        orig_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            xf = x.float().permute(0, 3, 2, 1).contiguous().view(B * T, F_b, C)
            xf_out, _ = self.gru_f(self.norm_f(xf))
            x_fp32 = x.float() + xf_out.view(B, T, F_b, C).permute(0, 3, 2, 1)

            xt = x_fp32.permute(0, 2, 3, 1).contiguous().view(B * F_b, T, C)
            xt_out, _ = self.gru_t(self.norm_t(xt))
            x_fp32 = x_fp32 + xt_out.view(B, F_b, T, C).permute(0, 3, 1, 2)
        return x_fp32.to(dtype=orig_dtype)


class _FullBinTFUNet(nn.Module):
    """Full-bin Frequency-U-Net (keeps full T) + Dual-Path BiGRU bottleneck."""

    def __init__(
        self,
        in_ch: int = 33,
        out_ch: int = 26,
        base_c: int = 32,
        n_bottleneck: int = 2,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        c0 = base_c
        c1 = base_c * 2
        c2 = base_c * 3
        c3 = base_c * 4
        c4 = base_c * 5

        self.in_conv = nn.Conv2d(in_ch, c0, kernel_size=3, padding=1, bias=False)
        self.enc0 = _ResBlock2D(c0)

        self.down1 = nn.Conv2d(c0, c1, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False)
        self.enc1 = _ResBlock2D(c1)

        self.down2 = nn.Conv2d(c1, c2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.enc2 = _ResBlock2D(c2)

        self.down3 = nn.Conv2d(c2, c3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.enc3 = _ResBlock2D(c3)

        self.down4 = nn.Conv2d(c3, c4, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False)
        self.bottleneck = nn.ModuleList([_DualPathBottleneck(c4) for _ in range(n_bottleneck)])

        self.up3 = nn.Conv2d(c4 + c3, c3, kernel_size=3, padding=1, bias=False)
        self.dec3 = _ResBlock2D(c3)

        self.up2 = nn.Conv2d(c3 + c2, c2, kernel_size=3, padding=1, bias=False)
        self.dec2 = _ResBlock2D(c2)

        self.up1 = nn.Conv2d(c2 + c1, c1, kernel_size=3, padding=1, bias=False)
        self.dec1 = _ResBlock2D(c1)

        self.up0 = nn.Conv2d(c1 + c0, c0, kernel_size=3, padding=1, bias=False)
        self.dec0 = _ResBlock2D(c0)

        self.out_norm = nn.GroupNorm(math.gcd(c0, 8), c0)
        self.out_conv = nn.Conv2d(c0, out_ch, kernel_size=1)
        # Near-zero init so model starts as a clean pass-through (zero cold-start shock)
        nn.init.normal_(self.out_conv.weight, mean=0.0, std=5e-3)
        nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def _up_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode="bilinear", align_corners=False)
        return torch.cat([x_up, skip], dim=1)

    def _run_enc_stage(self, down: nn.Module, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return block(down(x))

    def _run_dec_stage(self, up: nn.Module, block: nn.Module, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return block(up(self._up_cat(x, skip)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.enc0(self.in_conv(x))
        use_cp = self.training and self.grad_checkpoint and h0.requires_grad

        if use_cp:
            h1 = checkpoint(self._run_enc_stage, self.down1, self.enc1, h0, use_reentrant=False)
            h2 = checkpoint(self._run_enc_stage, self.down2, self.enc2, h1, use_reentrant=False)
            h3 = checkpoint(self._run_enc_stage, self.down3, self.enc3, h2, use_reentrant=False)
            h = self.down4(h3)
            for blk in self.bottleneck:
                h = checkpoint(blk, h, use_reentrant=False)
            h = checkpoint(self._run_dec_stage, self.up3, self.dec3, h, h3, use_reentrant=False)
            h = checkpoint(self._run_dec_stage, self.up2, self.dec2, h, h2, use_reentrant=False)
            h = checkpoint(self._run_dec_stage, self.up1, self.dec1, h, h1, use_reentrant=False)
            h = checkpoint(self._run_dec_stage, self.up0, self.dec0, h, h0, use_reentrant=False)
        else:
            h1 = self.enc1(self.down1(h0))
            h2 = self.enc2(self.down2(h1))
            h3 = self.enc3(self.down3(h2))
            h = self.down4(h3)
            for blk in self.bottleneck:
                h = blk(h)
            h = self.dec3(self.up3(self._up_cat(h, h3)))
            h = self.dec2(self.up2(self._up_cat(h, h2)))
            h = self.dec1(self.up1(self._up_cat(h, h1)))
            h = self.dec0(self.up0(self._up_cat(h, h0)))

        return self.out_conv(F.silu(self.out_norm(h)))


class CrossStemDebleedRefiner(nn.Module):
    """Chained full-bin 48 kHz 5-reference cross-stem residue canceller + 1D wave polish."""

    def __init__(
        self,
        sr: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 256,
        in_channels: int = 2,
        n_voices: int = 3,
        voice_base_c: int = 48,
        bg_base_c: int = 32,
        voice_bottleneck: int = 3,
        bg_bottleneck: int = 2,
        wave_hidden: int = 32,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        assert n_voices == 3, f"CrossStemDebleedRefiner v2 expects n_voices=3, got {n_voices}"
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.in_channels = in_channels
        self.n_voices = n_voices
        self.n_sources = n_voices + 2
        self.voice_base_c = voice_base_c
        self.bg_base_c = bg_base_c
        self.voice_bottleneck = voice_bottleneck
        self.bg_bottleneck = bg_bottleneck
        self.wave_hidden = wave_hidden
        self.grad_checkpoint = grad_checkpoint
        self.n_freqs = n_fft // 2 + 1

        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        # 6 complex signals (self, r1..r4, mix) * in_channels * 2 + 11 physics/coherence channels (+1 stem_flag for bg_net)
        in_ch_v = 6 * in_channels * 2 + 11
        in_ch_bg = in_ch_v + 1
        # Output channels: gate (C) + 6 complex filters (dM_self, M_r1, M_r2, M_r3, M_r4, M_mix) * 2*C = 13 * C
        out_ch = 13 * in_channels

        self.voice_net = _FullBinTFUNet(
            in_ch=in_ch_v,
            out_ch=out_ch,
            base_c=voice_base_c,
            n_bottleneck=voice_bottleneck,
            grad_checkpoint=grad_checkpoint,
        )
        self.bg_net = _FullBinTFUNet(
            in_ch=in_ch_bg,
            out_ch=out_ch,
            base_c=bg_base_c,
            n_bottleneck=bg_bottleneck,
            grad_checkpoint=grad_checkpoint,
        )

        # Tiny RMS-bounded 1D waveform de-grain polish for voices ([v_stft, v_other_sum, bg_sum, mix])
        self.voice_wave_net = nn.Sequential(
            nn.Conv1d(4, wave_hidden, kernel_size=15, padding=7, bias=False),
            nn.SiLU(),
            nn.Conv1d(wave_hidden, wave_hidden, kernel_size=15, padding=14, dilation=2, bias=False),
            nn.SiLU(),
            nn.Conv1d(wave_hidden, 1, kernel_size=15, padding=7, bias=False),
        )
        nn.init.normal_(self.voice_wave_net[-1].weight, mean=0.0, std=5e-3)

    def _stft(self, wav: torch.Tensor) -> torch.Tensor:
        lead = wav.shape[:-2]
        C, N = wav.shape[-2], wav.shape[-1]
        X = torch.stft(
            wav.reshape(-1, N).float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(device=wav.device, dtype=torch.float32),
            center=True,
            return_complex=True,
        )
        return X.reshape(*lead, C, self.n_freqs, X.shape[-1])

    def _istft(self, X: torch.Tensor, length: int) -> torch.Tensor:
        lead = X.shape[:-2]
        Xf = X.reshape(-1, self.n_freqs, X.shape[-1])
        wav = torch.istft(
            Xf,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(device=X.device, dtype=torch.float32),
            center=True,
            length=length,
        )
        return wav.reshape(*lead, length)

    @staticmethod
    def _compress_ri(Z: torch.Tensor) -> torch.Tensor:
        mag = Z.abs()
        sc = torch.log1p(mag) / (mag + 1e-6)
        return torch.cat([Z.real * sc, Z.imag * sc], dim=1)

    @staticmethod
    def _phase_coh(Ya: torch.Tensor, Yb: torch.Tensor) -> torch.Tensor:
        num = (Ya.real * Yb.real + Ya.imag * Yb.imag).mean(dim=1, keepdim=True)
        den = (Ya.abs() * Yb.abs()).mean(dim=1, keepdim=True) + 1e-5
        return (num / den).clamp(-1.0, 1.0)

    def _build_features(
        self,
        Y_self: torch.Tensor,
        Y_r1: torch.Tensor,
        Y_r2: torch.Tensor,
        Y_r3: torch.Tensor,
        Y_r4: torch.Tensor,
        Y_mix: torch.Tensor,
        scale: torch.Tensor,
        stem_flag: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build 35-ch (voice) or 36-ch (bg with stem_flag) physics & full 4-ref coherence tensor."""
        Ys = Y_self / scale
        Yr1 = Y_r1 / scale
        Yr2 = Y_r2 / scale
        Yr3 = Y_r3 / scale
        Yr4 = Y_r4 / scale
        Ymx = Y_mix / scale

        m_s = Ys.abs().mean(dim=1, keepdim=True)
        m_1 = Yr1.abs().mean(dim=1, keepdim=True)
        m_2 = Yr2.abs().mean(dim=1, keepdim=True)
        m_3 = Yr3.abs().mean(dim=1, keepdim=True)
        m_4 = Yr4.abs().mean(dim=1, keepdim=True)

        ratio_12 = m_s / (m_s + m_1 + m_2 + 1e-4)
        ratio_all = m_s / (m_s + m_1 + m_2 + m_3 + m_4 + 1e-4)
        coh_1 = self._phase_coh(Ys, Yr1)
        coh_2 = self._phase_coh(Ys, Yr2)
        coh_3 = self._phase_coh(Ys, Yr3)
        coh_4 = self._phase_coh(Ys, Yr4)

        parts = [
            self._compress_ri(Ys),
            self._compress_ri(Yr1),
            self._compress_ri(Yr2),
            self._compress_ri(Yr3),
            self._compress_ri(Yr4),
            self._compress_ri(Ymx),
            torch.log1p(m_s),
            torch.log1p(m_1),
            torch.log1p(m_2),
            torch.log1p(m_3),
            torch.log1p(m_4),
            ratio_12,
            ratio_all,
            coh_1,
            coh_2,
            coh_3,
            coh_4,
        ]
        if stem_flag is not None:
            parts.append(stem_flag.to(device=m_s.device, dtype=m_s.dtype).expand_as(m_s))
        return torch.cat(parts, dim=1)

    def _apply_cancel(
        self,
        net: nn.Module,
        Y_self: torch.Tensor,
        Y_r1: torch.Tensor,
        Y_r2: torch.Tensor,
        Y_r3: torch.Tensor,
        Y_r4: torch.Tensor,
        Y_mix: torch.Tensor,
        scale: torch.Tensor,
        stem_flag: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run full-bin U-Net and apply bounded 5-reference complex cancellation."""
        C = Y_self.shape[1]
        feats = self._build_features(Y_self, Y_r1, Y_r2, Y_r3, Y_r4, Y_mix, scale, stem_flag)
        raw = net(feats).float()  # [N, 13*C, F, T]

        g_raw = raw[:, 0:C]
        s_re, s_im = raw[:, C : 2 * C], raw[:, 2 * C : 3 * C]
        r1_re, r1_im = raw[:, 3 * C : 4 * C], raw[:, 4 * C : 5 * C]
        r2_re, r2_im = raw[:, 5 * C : 6 * C], raw[:, 6 * C : 7 * C]
        r3_re, r3_im = raw[:, 7 * C : 8 * C], raw[:, 8 * C : 9 * C]
        r4_re, r4_im = raw[:, 9 * C : 10 * C], raw[:, 10 * C : 11 * C]
        mx_re, mx_im = raw[:, 11 * C : 12 * C], raw[:, 12 * C : 13 * C]

        gate = torch.sigmoid(g_raw + 3.0)
        dM_self = torch.complex(torch.tanh(s_re), torch.tanh(s_im))
        M_r1 = torch.complex(torch.tanh(r1_re), torch.tanh(r1_im))
        M_r2 = torch.complex(torch.tanh(r2_re), torch.tanh(r2_im))
        M_r3 = torch.complex(torch.tanh(r3_re), torch.tanh(r3_im))
        M_r4 = torch.complex(torch.tanh(r4_re), torch.tanh(r4_im))
        M_mx = torch.complex(torch.tanh(mx_re), torch.tanh(mx_im))

        return (
            gate * (1.0 + 0.50 * dM_self) * Y_self
            - 0.85 * M_r1 * Y_r1
            - 0.85 * M_r2 * Y_r2
            - 0.85 * M_r3 * Y_r3
            - 0.85 * M_r4 * Y_r4
            + 0.35 * M_mx * Y_mix
        )

    def _run_wave_net(self, wave_in: torch.Tensor) -> torch.Tensor:
        return self.voice_wave_net(wave_in)

    def forward(
        self,
        raw_stems: torch.Tensor,
        mix: torch.Tensor,
        active_voice_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Refine 5 separated stems [B, 5, C, N] given original mix [B, C, N]."""
        B, S, C, N = raw_stems.shape
        V = self.n_voices
        assert S == V + 2, f"Expected {V + 2} stems, got {S}"

        Y_all = self._stft(raw_stems)  # [B, 5, C, F, T]
        Y_mix = self._stft(mix)        # [B, C, F, T]
        F_b, T_f = Y_all.shape[-2], Y_all.shape[-1]
        scale_b = Y_mix.abs().mean(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-5)  # [B, 1, 1, 1]

        # Per-voice RMS energy [B, V] for deterministic permutation-invariant reference ordering
        v_energy = raw_stems[:, :V].float().pow(2).mean(dim=(2, 3))  # [B, 3]
        Y_v = Y_all[:, :V]
        Y_music = Y_all[:, V]      # [B, C, F, T]
        Y_amb = Y_all[:, V + 1]    # [B, C, F, T]

        # Build sorted other-voice references (Y_o1 >= Y_o2) for each target voice v
        o1_list, o2_list = [], []
        for v in range(V):
            others = [u for u in range(V) if u != v]  # 2 indices
            ea = v_energy[:, others[0]]
            eb = v_energy[:, others[1]]
            swap = (eb > ea).view(B, 1, 1, 1)
            ya = Y_v[:, others[0]]
            yb = Y_v[:, others[1]]
            o1_list.append(torch.where(swap, yb, ya))
            o2_list.append(torch.where(swap, ya, yb))
        Y_o1 = torch.stack(o1_list, dim=1)  # [B, V, C, F, T]
        Y_o2 = torch.stack(o2_list, dim=1)  # [B, V, C, F, T]

        if not self.training:
            Y_v_ref_list = []
            for v in range(V):
                yv_out = self._apply_cancel(
                    self.voice_net,
                    Y_v[:, v],
                    Y_o1[:, v],
                    Y_o2[:, v],
                    Y_music,
                    Y_amb,
                    Y_mix,
                    scale_b,
                )
                Y_v_ref_list.append(yv_out)
            Y_v_ref = torch.stack(Y_v_ref_list, dim=1)
        else:
            Y_mus_v = Y_music.unsqueeze(1).expand(B, V, C, F_b, T_f)
            Y_amb_v = Y_amb.unsqueeze(1).expand(B, V, C, F_b, T_f)
            Y_mix_v = Y_mix.unsqueeze(1).expand(B, V, C, F_b, T_f)
            scale_v = scale_b.unsqueeze(1).expand(B, V, 1, 1, 1)
            Y_v_ref = self._apply_cancel(
                self.voice_net,
                Y_v.reshape(B * V, C, F_b, T_f),
                Y_o1.reshape(B * V, C, F_b, T_f),
                Y_o2.reshape(B * V, C, F_b, T_f),
                Y_mus_v.reshape(B * V, C, F_b, T_f),
                Y_amb_v.reshape(B * V, C, F_b, T_f),
                Y_mix_v.reshape(B * V, C, F_b, T_f),
                scale_v.reshape(B * V, 1, 1, 1),
            ).reshape(B, V, C, F_b, T_f)

        # Strict silent-slot mask
        if active_voice_mask is None:
            active_voice_mask = (v_energy.sqrt() > 1e-4).float()
        act_mask_f = active_voice_mask.to(device=Y_v_ref.device, dtype=torch.float32)
        Y_v_ref = Y_v_ref * act_mask_f.view(B, V, 1, 1, 1)

        # Refine Music (+1.0 flag) & Ambience (-1.0 flag) with all 3 voices sorted by descending energy
        v_order = torch.argsort(v_energy, dim=1, descending=True)  # [B, 3]
        b_idx = torch.arange(B, device=Y_v.device).unsqueeze(1)
        Y_v_sorted = Y_v[b_idx, v_order]  # [B, 3, C, F, T]
        Y_f = Y_all[:, V:]                # [B, 2, C, F, T]
        Y_f_other = Y_f.flip(dims=[1])    # [B, 2, C, F, T]

        if not self.training:
            Y_f_ref_list = []
            for s, flag_val in ((0, 1.0), (1, -1.0)):
                bg_flag = torch.full((B, 1, 1, 1), flag_val, device=Y_v.device, dtype=torch.float32)
                yf_out = self._apply_cancel(
                    self.bg_net,
                    Y_f[:, s],
                    Y_v_sorted[:, 0],
                    Y_v_sorted[:, 1],
                    Y_v_sorted[:, 2],
                    Y_f_other[:, s],
                    Y_mix,
                    scale_b,
                    bg_flag,
                )
                Y_f_ref_list.append(yf_out)
            Y_f_ref = torch.stack(Y_f_ref_list, dim=1)
        else:
            bg_flag = (
                torch.tensor([1.0, -1.0], device=Y_v.device, dtype=torch.float32)
                .view(1, 2, 1, 1, 1)
                .expand(B, 2, 1, 1, 1)
                .reshape(B * 2, 1, 1, 1)
            )
            Y_vs0 = Y_v_sorted[:, 0:1].expand(B, 2, C, F_b, T_f)
            Y_vs1 = Y_v_sorted[:, 1:2].expand(B, 2, C, F_b, T_f)
            Y_vs2 = Y_v_sorted[:, 2:3].expand(B, 2, C, F_b, T_f)
            Y_mix_f = Y_mix.unsqueeze(1).expand(B, 2, C, F_b, T_f)
            scale_f = scale_b.unsqueeze(1).expand(B, 2, 1, 1, 1)
            Y_f_ref = self._apply_cancel(
                self.bg_net,
                Y_f.reshape(B * 2, C, F_b, T_f),
                Y_vs0.reshape(B * 2, C, F_b, T_f),
                Y_vs1.reshape(B * 2, C, F_b, T_f),
                Y_vs2.reshape(B * 2, C, F_b, T_f),
                Y_f_other.reshape(B * 2, C, F_b, T_f),
                Y_mix_f.reshape(B * 2, C, F_b, T_f),
                scale_f.reshape(B * 2, 1, 1, 1),
                bg_flag,
            ).reshape(B, 2, C, F_b, T_f)

        # iSTFT -> [B, 5, C, N]
        stems_stft = self._istft(torch.cat([Y_v_ref, Y_f_ref], dim=1), length=N)

        # Tiny RMS-bounded 1D waveform de-grain polish on voices
        v_stft = stems_stft[:, :V]
        v_other_w = v_stft.sum(dim=1, keepdim=True) - v_stft
        bg_sum_w = stems_stft[:, V : V + 1] + stems_stft[:, V + 1 : V + 2]
        mix_f = mix.float()

        if not self.training:
            v_fin_list = []
            for v in range(V):
                w_in = torch.stack(
                    [v_stft[:, v], v_other_w[:, v], bg_sum_w[:, 0], mix_f], dim=2
                ).reshape(B * C, 4, N)
                raw_dw = self.voice_wave_net(w_in).reshape(B, C, N).float()
                vf = _apply_bounded_wave_residual(v_stft[:, v], raw_dw, max_ratio=0.15)
                v_fin_list.append(vf)
            v_final = torch.stack(v_fin_list, dim=1)
        else:
            bg_exp = bg_sum_w.expand(B, V, C, N)
            mx_exp = mix_f.unsqueeze(1).expand(B, V, C, N)
            wave_in = torch.stack([v_stft, v_other_w, bg_exp, mx_exp], dim=3).reshape(B * V * C, 4, N)
            if self.grad_checkpoint and wave_in.requires_grad:
                raw_dw = checkpoint(self._run_wave_net, wave_in, use_reentrant=False).reshape(B, V, C, N).float()
            else:
                raw_dw = self.voice_wave_net(wave_in).reshape(B, V, C, N).float()
            v_final = _apply_bounded_wave_residual(v_stft, raw_dw, max_ratio=0.15)

        v_final = v_final * act_mask_f.view(B, V, 1, 1)
        return {"stems": torch.cat([v_final, stems_stft[:, V:]], dim=1)}
