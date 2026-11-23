"""HarmonicSlot-TF: Multi-stem separator for 2-3 Voices, Music, and Ambience.

Supports 3 apples-to-apples ablation modes (`mode`):
  - "plain":          Dual-path band-split encoder -> linear stem projection.
  - "queries":        SepTDA-style competitive speaker slots + semantic queries +
                      inter-source attention (no purity gate).
  - "purity_queries": Same as "queries", plus decoupled 2-head purity gate
                      (class ratios [speech, music, ambience] + single-speaker
                      dominance among voices) biasing slot attention.
"""
from __future__ import annotations

import math
from typing import Dict, List, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

AblationMode = Literal["plain", "queries", "purity_queries"]


def build_variable_bands(
    n_freqs: int,
    sr: int = 48000,
    n_fft: int = 2048,
) -> List[Tuple[int, int]]:
    """Partition all STFT bins [0, n_freqs) with finer grouping in voice range."""
    hz_per_bin = sr / n_fft
    bands: List[Tuple[int, int]] = []
    i = 0
    while i < n_freqs:
        hz = i * hz_per_bin
        if hz < 1500:
            w = 2
        elif hz < 4500:
            w = 6
        elif hz < 10000:
            w = 24
        else:
            w = 64
        j = min(n_freqs, i + w)
        bands.append((i, j))
        i = j
    return bands


class HarmonicStackGather(nn.Module):
    """Gather integer harmonic multiples {1f..H*f} from STFT physics channels."""

    def __init__(
        self,
        n_freqs: int,
        sr: int = 48000,
        n_fft: int = 2048,
        max_f0_hz: float = 1200.0,
        n_harmonics: int = 6,
        out_dim: int = 64,
    ):
        super().__init__()
        hz_per_bin = sr / n_fft
        self.n_low = min(n_freqs, int(math.ceil(max_f0_hz / hz_per_bin)))
        self.n_harmonics = n_harmonics

        base = torch.arange(self.n_low, dtype=torch.long)
        mults = torch.arange(1, n_harmonics + 1, dtype=torch.long)
        harm_idx = base[:, None] * mults[None, :]
        valid = (harm_idx < n_freqs).float()
        self.register_buffer("harm_idx", harm_idx.clamp(max=n_freqs - 1))
        self.register_buffer("valid", valid)

        self.proj = nn.Sequential(
            nn.Linear(n_harmonics * 4, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        mag: torch.Tensor,
        if_dev: torch.Tensor,
        ild: torch.Tensor,
        ipd_cos: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = mag.shape
        feat = torch.stack([torch.log1p(mag), if_dev, ild, ipd_cos], dim=-1)
        flat_idx = self.harm_idx.reshape(-1)
        gathered = feat[:, :, flat_idx, :].reshape(B, T, self.n_low, self.n_harmonics, 4)
        gathered = gathered * self.valid.view(1, 1, self.n_low, self.n_harmonics, 1)
        return self.proj(gathered.reshape(B, T, self.n_low, self.n_harmonics * 4))


def _mha_chunked(
    mha: nn.MultiheadAttention,
    x: torch.Tensor,
    chunk_size: int = 16384,
) -> torch.Tensor:
    """Run batch-first self-attention in chunks so batch <= 65535 for CUDA SDPA kernels."""
    cs = min(chunk_size, 1024) if not mha.training else chunk_size
    if x.shape[0] <= cs:
        out, _ = mha(x, x, x, need_weights=False)
        return out
    return torch.cat(
        [mha(c, c, c, need_weights=False)[0] for c in x.split(cs, dim=0)],
        dim=0,
    )


class _RoPESelfAttention(nn.Module):
    """Multi-head self-attention with 1D Rotary Position Embeddings (RoPE) over sequence dim."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        theta: float = 10000.0,
        chunk_size: int = 16384,
    ):
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} must be divisible by n_heads={n_heads}"
        head_dim = dim // n_heads
        assert head_dim % 2 == 0, f"head_dim={head_dim} must be even for RoPE"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        t1, t2 = t.float().chunk(2, dim=-1)
        out = torch.cat([t1 * cos - t2 * sin, t2 * cos + t1 * sin], dim=-1)
        return out.to(dtype=t.dtype)

    def _run_chunk(self, c: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        N_c, L, D = c.shape
        qkv = self.qkv(c).view(N_c, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = self._apply_rope(qkv[0], cos, sin)
        k = self._apply_rope(qkv[1], cos, sin)
        v = qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(y.transpose(1, 2).reshape(N_c, L, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N_batch, L, _ = x.shape
        pos = torch.arange(L, device=x.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(device=x.device))
        cos = freqs.cos().view(1, 1, L, self.head_dim // 2)
        sin = freqs.sin().view(1, 1, L, self.head_dim // 2)
        cs = min(self.chunk_size, 256) if not self.training else self.chunk_size
        if N_batch <= cs:
            return self._run_chunk(x, cos, sin)
        return torch.cat([self._run_chunk(c, cos, sin) for c in x.split(cs, dim=0)], dim=0)


class DualPathBandBlock(nn.Module):
    """Band-axis Self-Attention + Time-axis BiGRU + optional global frame & inter-source attention."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        use_time_attn: bool = False,
        use_rope: bool = False,
        use_inter_source: bool = False,
    ):
        super().__init__()
        self.use_time_attn = use_time_attn
        self.use_rope = use_rope
        self.use_inter_source = use_inter_source
        self.norm_f = nn.LayerNorm(dim)
        self.attn_f = (
            _RoPESelfAttention(dim, n_heads=n_heads, dropout=dropout)
            if use_rope
            else nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        )

        self.norm_t = nn.LayerNorm(dim)
        self.gru_t = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)

        if use_time_attn:
            self.norm_t_attn = nn.LayerNorm(dim)
            self.attn_t = (
                _RoPESelfAttention(dim, n_heads=n_heads, dropout=dropout)
                if use_rope
                else nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
            )

        if use_inter_source:
            self.norm_src = nn.LayerNorm(dim)
            self.attn_src = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

        self.norm_mlp = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor, n_sources: int = 1) -> torch.Tensor:
        B, T, K, D = x.shape
        if not self.training:
            t_chunk = 128
            for b in range(B):
                xb = x[b : b + 1]
                xf = self.norm_f(xb).reshape(T, K, D)
                xf = self.attn_f(xf) if self.use_rope else _mha_chunked(self.attn_f, xf)
                out_b = xb + xf.reshape(1, T, K, D)
                if b == 0 and out_b.dtype != x.dtype:
                    x = x.to(dtype=out_b.dtype)
                x[b : b + 1] = out_b
                del xf, out_b

                xb = x[b : b + 1]
                xt = self.norm_t(xb).permute(0, 2, 1, 3).reshape(K, T, D)
                xt, _ = self.gru_t(xt)
                out_b = xb + xt.reshape(1, K, T, D).permute(0, 2, 1, 3)
                if b == 0 and out_b.dtype != x.dtype:
                    x = x.to(dtype=out_b.dtype)
                x[b : b + 1] = out_b
                del xt, out_b

            if self.use_time_attn:
                x_frame = self.norm_t_attn(x.mean(dim=2))
                xt_global = self.attn_t(x_frame) if self.use_rope else _mha_chunked(self.attn_t, x_frame)
                x = x + xt_global.unsqueeze(2)
                del x_frame, xt_global

            if self.use_inter_source and n_sources > 1:
                B_orig = B // n_sources
                xv = x.view(B_orig, n_sources, T, K, D)
                for t0 in range(0, T, t_chunk):
                    t1 = min(T, t0 + t_chunk)
                    tc = t1 - t0
                    xs = (
                        self.norm_src(xv[:, :, t0:t1])
                        .permute(0, 2, 3, 1, 4)
                        .reshape(B_orig * tc * K, n_sources, D)
                    )
                    xs = _mha_chunked(self.attn_src, xs)
                    out_s = xv[:, :, t0:t1] + (
                        xs.view(B_orig, tc, K, n_sources, D).permute(0, 3, 1, 2, 4)
                    )
                    if t0 == 0 and out_s.dtype != x.dtype:
                        x = x.to(dtype=out_s.dtype)
                        xv = x.view(B_orig, n_sources, T, K, D)
                    xv[:, :, t0:t1] = out_s
                    del xs, out_s

            for b in range(B):
                for t0 in range(0, T, t_chunk):
                    t1 = min(T, t0 + t_chunk)
                    xb = x[b : b + 1, t0:t1]
                    out_m = xb + self.ffn(self.norm_mlp(xb))
                    if b == 0 and t0 == 0 and out_m.dtype != x.dtype:
                        x = x.to(dtype=out_m.dtype)
                    x[b : b + 1, t0:t1] = out_m
                    del out_m
            return x

        xf = self.norm_f(x).reshape(B * T, K, D)
        xf = self.attn_f(xf) if self.use_rope else _mha_chunked(self.attn_f, xf)
        x = x + xf.reshape(B, T, K, D)

        xt = self.norm_t(x).permute(0, 2, 1, 3).reshape(B * K, T, D)
        xt, _ = self.gru_t(xt)
        x = x + xt.reshape(B, K, T, D).permute(0, 2, 1, 3)

        if self.use_time_attn:
            x_frame = self.norm_t_attn(x.mean(dim=2))
            xt_global = self.attn_t(x_frame) if self.use_rope else _mha_chunked(self.attn_t, x_frame)
            x = x + xt_global.unsqueeze(2)

        if self.use_inter_source and n_sources > 1:
            B_orig = B // n_sources
            xs = (
                self.norm_src(x)
                .view(B_orig, n_sources, T, K, D)
                .permute(0, 2, 3, 1, 4)
                .reshape(B_orig * T * K, n_sources, D)
            )
            xs = _mha_chunked(self.attn_src, xs)
            x = x + (
                xs.view(B_orig, T, K, n_sources, D)
                .permute(0, 3, 1, 2, 4)
                .reshape(B, T, K, D)
            )

        x = x + self.ffn(self.norm_mlp(x))
        return x


class SlotQueryModule(nn.Module):
    """Competitive voice slots + semantic Music/Ambience queries + optional purity gate."""

    def __init__(
        self,
        dim: int,
        n_voices: int = 3,
        n_iters: int = 3,
        n_heads: int = 4,
        use_purity: bool = True,
        legacy_slots: bool = False,
        arch_version: str = "v2",
    ):
        super().__init__()
        self.n_voices = n_voices
        self.n_sources = n_voices + 2
        self.n_iters = n_iters
        self.use_purity = use_purity
        self.legacy_slots = legacy_slots
        self.arch_version = arch_version
        self.scale = dim ** -0.5

        if use_purity:
            # Head 1: [speech, music, ambience] class ratio logits
            self.class_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))
            # Head 2: single-speaker dominance logit among voices
            self.solo_spk_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

        if legacy_slots:
            # Legacy Phase 1: shared Gaussian distribution for voice slots
            self.voice_mu = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.voice_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        else:
            # v2/v3: distinct orthogonal learnable queries (uPIT provides permutation invariance)
            q_init = torch.empty(n_voices, dim)
            nn.init.orthogonal_(q_init, gain=1.0)
            self.voice_queries = nn.Parameter(q_init.unsqueeze(0))  # [1, V, D]

        # Fixed semantic queries for Music and Ambience
        self.fixed_queries = nn.Parameter(torch.randn(1, 2, dim) * 0.02)

        self.norm_x = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        if not legacy_slots:
            if arch_version == "v3":
                self.voice_stem_proj = nn.Linear(dim, dim)
                self.fixed_stem_proj = nn.Linear(dim, 2 * dim)
            else:
                self.stem_proj = nn.Linear(dim, self.n_sources * dim)

        # FiLM conditioning from each source slot into the (T, K) grid
        self.film_gamma = nn.Linear(dim, dim)
        self.film_beta = nn.Linear(dim, dim)

        # Inter-source self-attention across the S source axis at each (t, k)
        self.norm_inter = nn.LayerNorm(dim)
        self.inter_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

        if arch_version == "v3":
            self.prev_gate = nn.Linear(dim * 2, dim)
            nn.init.constant_(self.prev_gate.bias, 2.0)
            self.norm_inter2 = nn.LayerNorm(dim)
            self.inter_attn2 = nn.MultiheadAttention(dim, n_heads, batch_first=True)
            self.norm_inter_ffn = nn.LayerNorm(dim)
            self.inter_ffn = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim),
            )

    def forward(
        self, x: torch.Tensor, prev_slots: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        x: [B, T, K, D]
        prev_slots: optional [B, V, D] carried voiceprints from previous chunk (v3)
        Returns:
          slot_ctx:        [B, S, T, K, D]
          slots:           [B, S, D]
          class_logits:    [B, T, K, 3] or None
          solo_spk_logits: [B, T, K, 1] or None
        """
        B, T, K, D = x.shape
        N = T * K
        x_flat = self.norm_x(x).reshape(B, N, D)
        k_tok = self.k_proj(x_flat)  # [B, N, D]
        v_tok = self.v_proj(x_flat)  # [B, N, D]

        class_logits = None
        solo_spk_logits = None
        cls_probs = None
        p_voice = None
        slot_bias = None
        if self.use_purity:
            class_logits = self.class_head(x)           # [B, T, K, 3]
            solo_spk_logits = self.solo_spk_head(x)     # [B, T, K, 1]
            cls_probs = F.softmax(class_logits.float(), dim=-1).reshape(B, N, 3)
            solo_probs = torch.sigmoid(solo_spk_logits.float()).reshape(B, N, 1)
            # Token-level single-speaker weight in (0, 1] applied across N after slot competition
            p_voice = (cls_probs[:, :, 0:1] * solo_probs).permute(0, 2, 1)  # [B, 1, N]
            voice_bias = torch.log(cls_probs[:, :, 0:1].clamp_min(1e-5))    # [B, N, 1]
            music_bias = torch.log(cls_probs[:, :, 1:2].clamp_min(1e-5))    # [B, N, 1]
            amb_bias = torch.log(cls_probs[:, :, 2:3].clamp_min(1e-5))      # [B, N, 1]
            slot_bias = torch.cat(
                [voice_bias.expand(B, N, self.n_voices), music_bias, amb_bias], dim=-1
            ).permute(0, 2, 1)  # [B, S, N]

        if self.arch_version == "v3":
            base_q = self.voice_queries.expand(B, -1, -1).to(dtype=x.dtype)
            if prev_slots is not None:
                ps = prev_slots.to(device=x.device, dtype=x.dtype)
                bq_norm = base_q.float().norm(dim=-1, keepdim=True).clamp_min(1e-4)
                ps_norm = ps.float().norm(dim=-1, keepdim=True).clamp_min(1e-4)
                ps = (ps.float() * (bq_norm / ps_norm)).to(dtype=x.dtype)
                g = torch.sigmoid(self.prev_gate(torch.cat([ps, base_q], dim=-1)))
                v_slots = g * ps + (1.0 - g) * base_q
            else:
                v_slots = base_q
        elif self.legacy_slots:
            # Sample distinct voice slots symmetrically without mutating global RNG in eval
            if self.training:
                eps = torch.randn(B, self.n_voices, D, device=x.device, dtype=x.dtype)
            else:
                gen = torch.Generator(device=x.device)
                gen.manual_seed(0)
                eps = torch.randn(
                    1, self.n_voices, D, device=x.device, dtype=x.dtype, generator=gen
                ).expand(B, -1, -1)
            v_slots = self.voice_mu + self.voice_log_sigma.exp() * eps
        else:
            v_slots = self.voice_queries.expand(B, -1, -1)
        f_slots = self.fixed_queries.expand(B, -1, -1).to(dtype=x.dtype)
        slots = torch.cat([v_slots, f_slots], dim=1)  # [B, S, D]

        # Iterative competitive slot attention across all S = V + 2 slots
        comp = None
        for _ in range(self.n_iters):
            slots_prev = slots
            q = self.q_proj(self.norm_slots(slots))  # [B, S, D]
            logits = (torch.einsum("bsd,bnd->bsn", q, k_tok) * self.scale).float()
            if slot_bias is not None:
                logits = logits + slot_bias

            comp = torch.softmax(logits, dim=1)  # compete across all S sources (voices + music + ambience)
            if self.arch_version == "v3":
                # Use solo-speaker probability ONLY to pool clean voiceprints; keep full overlap conditioning below
                if p_voice is not None:
                    pool_w = torch.cat([comp[:, : self.n_voices, :] * p_voice, comp[:, self.n_voices :, :]], dim=1)
                else:
                    pool_w = comp
                attn = (pool_w / (pool_w.sum(dim=-1, keepdim=True) + 1e-6)).to(dtype=v_tok.dtype)
            else:
                if p_voice is not None:
                    v_weighted = comp[:, : self.n_voices, :] * p_voice
                    comp = torch.cat([v_weighted, comp[:, self.n_voices :, :]], dim=1)
                attn = (comp / (comp.sum(dim=-1, keepdim=True) + 1e-6)).to(dtype=v_tok.dtype)  # [B, S, N]

            updates = torch.einsum("bsn,bnd->bsd", attn, v_tok)
            if not self.legacy_slots:
                u_v = updates[:, : self.n_voices]
                u_v = u_v - u_v.mean(dim=1, keepdim=True)
                updates = torch.cat([u_v, updates[:, self.n_voices :]], dim=1)
            slots = self.gru(
                updates.reshape(B * self.n_sources, D),
                slots_prev.reshape(B * self.n_sources, D),
            ).reshape(B, self.n_sources, D)
            slots = slots + self.slot_mlp(slots)
            if self.arch_version == "v3":
                s_v = slots[:, : self.n_voices]
                s_v = (s_v - s_v.mean(dim=1, keepdim=True)) + v_slots
                slots = torch.cat([s_v, slots[:, self.n_voices :]], dim=1)
            elif not self.legacy_slots:
                s_v = slots[:, : self.n_voices]
                s_v = (s_v - s_v.mean(dim=1, keepdim=True)) + v_slots
                slots = torch.cat([s_v, slots[:, self.n_voices :]], dim=1)

        # Condition (T, K) representation on each extracted source slot via FiLM (+ per-source subspace)
        gamma = self.film_gamma(slots).view(B, self.n_sources, 1, 1, D)
        beta = self.film_beta(slots).view(B, self.n_sources, 1, 1, D)
        if not self.training:
            del x_flat, v_tok, p_voice
            q_fin = self.q_proj(self.norm_slots(slots)) if self.arch_version == "v3" else None
            k_btk = k_tok.view(B, T, K, D)
            sb_btk = slot_bias.view(B, self.n_sources, T, K) if slot_bias is not None else None
            cls_btk = cls_probs.view(B, T, K, 3) if cls_probs is not None else None
            comp_btk = comp.view(B, self.n_sources, T, K) if comp is not None else None
            slot_ctx: torch.Tensor | None = None
            t_chunk = 128
            for t0 in range(0, T, t_chunk):
                t1 = min(T, t0 + t_chunk)
                tc = t1 - t0
                nc = tc * K
                xc = x[:, t0:t1]
                if self.arch_version == "v3":
                    assert q_fin is not None
                    kc = k_btk[:, t0:t1].reshape(B, nc, D)
                    logits_c = (torch.einsum("bsd,bnd->bsn", q_fin, kc) * self.scale).float()
                    if sb_btk is not None:
                        logits_c = logits_c + sb_btk[:, :, t0:t1].reshape(B, self.n_sources, nc)
                    cond_c = torch.softmax(logits_c, dim=1).to(dtype=xc.dtype).view(B, self.n_sources, tc, K, 1)
                    x_v = self.voice_stem_proj(xc).unsqueeze(1).expand(B, self.n_voices, tc, K, D)
                    x_f = self.fixed_stem_proj(xc).view(B, tc, K, 2, D).permute(0, 3, 1, 2, 4)
                    x_s = torch.cat([x_v, x_f], dim=1)
                    if self.use_purity:
                        assert cls_btk is not None
                        clsc = cls_btk[:, t0:t1].to(dtype=xc.dtype)
                        speech_base = 0.5 * clsc[..., 0:1].unsqueeze(1).expand(B, self.n_voices, tc, K, 1)
                        fixed_base = 0.5 * clsc[..., 1:3].permute(0, 3, 1, 2).unsqueeze(-1)
                        base_c = torch.cat([speech_base, fixed_base], dim=1)
                    else:
                        base_c = 0.5
                    scale_c = base_c + cond_c
                    sc = x_s * (1.0 + gamma * scale_c) + beta * scale_c
                elif self.legacy_slots:
                    sc = xc.unsqueeze(1) * (1.0 + gamma) + beta
                else:
                    assert comp_btk is not None
                    x_s = self.stem_proj(xc).view(B, tc, K, self.n_sources, D).permute(0, 3, 1, 2, 4)
                    comp_c = comp_btk[:, :, t0:t1].to(dtype=xc.dtype).unsqueeze(-1)
                    sc = x_s * (1.0 + gamma * comp_c) + beta * comp_c

                inter_in = self.norm_inter(sc).permute(0, 2, 3, 1, 4).reshape(B * nc, self.n_sources, D)
                inter_out = _mha_chunked(self.inter_attn, inter_in)
                sc = sc + inter_out.reshape(B, tc, K, self.n_sources, D).permute(0, 3, 1, 2, 4)
                del inter_in, inter_out

                if self.arch_version == "v3":
                    inter_in2 = self.norm_inter2(sc).permute(0, 2, 3, 1, 4).reshape(B * nc, self.n_sources, D)
                    inter_out2 = _mha_chunked(self.inter_attn2, inter_in2)
                    sc = sc + inter_out2.reshape(B, tc, K, self.n_sources, D).permute(0, 3, 1, 2, 4)
                    del inter_in2, inter_out2
                    sc = sc + self.inter_ffn(self.norm_inter_ffn(sc))

                if slot_ctx is None:
                    slot_ctx = torch.empty(B, self.n_sources, T, K, D, device=sc.device, dtype=sc.dtype)
                slot_ctx[:, :, t0:t1] = sc
                del sc
            assert slot_ctx is not None
            return slot_ctx, slots, class_logits, solo_spk_logits

        if self.arch_version == "v3":
            q_fin = self.q_proj(self.norm_slots(slots))
            logits_fin = (torch.einsum("bsd,bnd->bsn", q_fin, k_tok) * self.scale).float()
            if slot_bias is not None:
                logits_fin = logits_fin + slot_bias
            cond_tk = torch.softmax(logits_fin, dim=1).to(dtype=x.dtype).view(B, self.n_sources, T, K, 1)
            x_v = self.voice_stem_proj(x).unsqueeze(1).expand(B, self.n_voices, T, K, D)
            x_f = self.fixed_stem_proj(x).view(B, T, K, 2, D).permute(0, 3, 1, 2, 4)
            x_s = torch.cat([x_v, x_f], dim=1)
            if self.use_purity:
                assert cls_probs is not None
                speech_base = (
                    0.5
                    * cls_probs[:, :, 0:1].to(dtype=x.dtype).view(B, 1, T, K, 1).expand(B, self.n_voices, T, K, 1)
                )
                fixed_base = 0.5 * cls_probs[:, :, 1:3].to(dtype=x.dtype).permute(0, 2, 1).view(B, 2, T, K, 1)
                base_tk = torch.cat([speech_base, fixed_base], dim=1)
            else:
                base_tk = 0.5
            scale_tk = base_tk + cond_tk
            slot_ctx = x_s * (1.0 + gamma * scale_tk) + beta * scale_tk
        elif self.legacy_slots:
            slot_ctx = x.unsqueeze(1) * (1.0 + gamma) + beta  # [B, S, T, K, D]
        else:
            assert comp is not None
            x_s = self.stem_proj(x).view(B, T, K, self.n_sources, D).permute(0, 3, 1, 2, 4)
            comp_tk = comp.to(dtype=x.dtype).view(B, self.n_sources, T, K, 1)
            slot_ctx = x_s * (1.0 + gamma * comp_tk) + beta * comp_tk  # [B, S, T, K, D]

        # Inter-source self-attention across S at every (t, k)
        inter_in = self.norm_inter(slot_ctx).permute(0, 2, 3, 1, 4).reshape(B * N, self.n_sources, D)
        inter_out = _mha_chunked(self.inter_attn, inter_in)
        slot_ctx = slot_ctx + inter_out.reshape(B, T, K, self.n_sources, D).permute(0, 3, 1, 2, 4)

        if self.arch_version == "v3":
            inter_in2 = self.norm_inter2(slot_ctx).permute(0, 2, 3, 1, 4).reshape(B * N, self.n_sources, D)
            inter_out2 = _mha_chunked(self.inter_attn2, inter_in2)
            slot_ctx = slot_ctx + inter_out2.reshape(B, T, K, self.n_sources, D).permute(0, 3, 1, 2, 4)
            slot_ctx = slot_ctx + self.inter_ffn(self.norm_inter_ffn(slot_ctx))

        return slot_ctx, slots, class_logits, solo_spk_logits


def _apply_bounded_residual(
    base: torch.Tensor,
    raw_delta: torch.Tensor,
    mask: torch.Tensor | None = None,
    max_ratio: float = 0.20,
) -> torch.Tensor:
    """Apply an RMS-bounded waveform residual with any self-canceling negative projection stripped."""
    rms = base.pow(2).mean(dim=(-2, -1), keepdim=True).add(1e-12).sqrt()
    delta = max_ratio * rms * torch.tanh(raw_delta.float())
    if mask is not None:
        delta = delta * mask
    proj = (delta * base).sum(dim=(-2, -1), keepdim=True) / (
        base.pow(2).sum(dim=(-2, -1), keepdim=True) + 1e-8
    )
    return base + delta - F.relu(-proj) * base


class CrossResidueEnhancer(nn.Module):
    """2nd-pass cross-stem STFT + waveform residue canceller using other separated stems as references."""

    def __init__(self, hidden_stft: int = 16, hidden_wave: int = 16, apply_gate: bool = False):
        super().__init__()
        self.apply_gate = apply_gate
        self.voice_stft_net = nn.Sequential(
            nn.Conv2d(9, hidden_stft, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden_stft, hidden_stft, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden_stft, 6, 3, padding=1, bias=False),
        )
        self.bg_stft_net = nn.Sequential(
            nn.Conv2d(9, hidden_stft, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden_stft, 6, 3, padding=1, bias=False),
        )
        self.voice_wave_net = nn.Sequential(
            nn.Conv1d(3, hidden_wave, 15, padding=7, bias=False),
            nn.SiLU(),
            nn.Conv1d(hidden_wave, hidden_wave, 15, padding=14, dilation=2, bias=False),
            nn.SiLU(),
            nn.Conv1d(hidden_wave, 1, 15, padding=7, bias=False),
        )
        for net in (self.voice_stft_net, self.bg_stft_net, self.voice_wave_net):
            nn.init.normal_(net[-1].weight, mean=0.0, std=1e-3)

    @staticmethod
    def _build_stft_feats(
        Y_self: torch.Tensor, Y_ref1: torch.Tensor, Y_ref2: torch.Tensor
    ) -> torch.Tensor:
        scale = (Y_self.abs() + Y_ref1.abs() + Y_ref2.abs()).mean(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        Ys = Y_self / scale
        Yr1 = Y_ref1 / scale
        Yr2 = Y_ref2 / scale
        return torch.stack(
            [
                Ys.real,
                Ys.imag,
                torch.log1p(Ys.abs()),
                Yr1.real,
                Yr1.imag,
                torch.log1p(Yr1.abs()),
                Yr2.real,
                Yr2.imag,
                torch.log1p(Yr2.abs()),
            ],
            dim=1,
        )

    def _apply_stft_cancel(
        self,
        net: nn.Module,
        Y_self: torch.Tensor,
        Y_ref1: torch.Tensor,
        Y_ref2: torch.Tensor,
    ) -> torch.Tensor:
        def _run_net(Y_s: torch.Tensor, Y_r1: torch.Tensor, Y_r2: torch.Tensor) -> torch.Tensor:
            feats = self._build_stft_feats(Y_s, Y_r1, Y_r2)
            return net(feats).float()

        if self.training and (Y_self.requires_grad or Y_ref1.requires_grad or Y_ref2.requires_grad):
            raw = checkpoint(_run_net, Y_self, Y_ref1, Y_ref2, use_reentrant=False)
        else:
            raw = _run_net(Y_self, Y_ref1, Y_ref2)
        dM_self = torch.complex(torch.tanh(raw[:, 0]), torch.tanh(raw[:, 1]))
        M_c1 = torch.complex(torch.tanh(raw[:, 2]), torch.tanh(raw[:, 3]))
        M_c2 = torch.complex(torch.tanh(raw[:, 4]), torch.tanh(raw[:, 5]))
        return (1.0 + 0.15 * dM_self) * Y_self - 0.25 * M_c1 * Y_ref1 - 0.25 * M_c2 * Y_ref2

    def forward(
        self,
        stems: torch.Tensor,
        gate: torch.Tensor,
        stft_fn,
        istft_fn,
    ) -> torch.Tensor:
        B, S, C, N = stems.shape
        V = S - 2
        Y_all = stft_fn(stems.float().reshape(B * S, C, N))
        _, _, F_bins, T_frames = Y_all.shape
        Y_all = Y_all.reshape(B, S, C, F_bins, T_frames)

        Y_v = Y_all[:, :V]
        Y_v_sum = Y_v.sum(dim=1, keepdim=True)
        Y_bg_sum_1 = Y_all[:, V : V + 1] + Y_all[:, V + 1 : V + 2]
        if not self.training:
            g_v = gate[:, :V].float() if self.apply_gate else None
            Y_v_list = []
            for v in range(V):
                yv_self = Y_v[:, v].reshape(B * C, F_bins, T_frames)
                yv_other = (Y_v_sum[:, 0] - Y_v[:, v]).reshape(B * C, F_bins, T_frames)
                yv_bg = Y_bg_sum_1[:, 0].reshape(B * C, F_bins, T_frames)
                yv_e = self._apply_stft_cancel(self.voice_stft_net, yv_self, yv_other, yv_bg).reshape(
                    B, 1, C, F_bins, T_frames
                )
                if g_v is not None:
                    yv_e = yv_e * g_v[:, v : v + 1].view(B, 1, 1, 1, 1)
                Y_v_list.append(yv_e)
            Y_v_enh = torch.cat(Y_v_list, dim=1)
            del Y_v_list, Y_v, Y_bg_sum_1

            Y_f = Y_all[:, V:]
            Y_f_list = []
            for fi in range(2):
                yf_self = Y_f[:, fi].reshape(B * C, F_bins, T_frames)
                yf_other = Y_f[:, 1 - fi].reshape(B * C, F_bins, T_frames)
                yf_v = Y_v_sum[:, 0].reshape(B * C, F_bins, T_frames)
                Y_f_list.append(
                    self._apply_stft_cancel(self.bg_stft_net, yf_self, yf_other, yf_v).reshape(
                        B, 1, C, F_bins, T_frames
                    )
                )
            Y_f_enh = torch.cat(Y_f_list, dim=1)
            del Y_f_list, Y_f, Y_v_sum, Y_all

            stems_stft = istft_fn(torch.cat([Y_v_enh, Y_f_enh], dim=1), length=N)
            del Y_v_enh, Y_f_enh
            v_enh = stems_stft[:, :V]
            v_sum_w = v_enh.sum(dim=1, keepdim=True)
            v_bg_w1 = stems_stft[:, V : V + 1] + stems_stft[:, V + 1 : V + 2]
            v_fin_list = []
            for v in range(V):
                w_in = torch.stack(
                    [v_enh[:, v], v_sum_w[:, 0] - v_enh[:, v], v_bg_w1[:, 0]], dim=2
                ).reshape(B * C, 3, N)
                raw_dw = self.voice_wave_net(w_in).reshape(B, 1, C, N).float()
                vf = _apply_bounded_residual(v_enh[:, v : v + 1], raw_dw)
                if g_v is not None:
                    vf = vf * g_v[:, v : v + 1].view(B, 1, 1, 1)
                v_fin_list.append(vf)
                del w_in, raw_dw
            return torch.cat([torch.cat(v_fin_list, dim=1), stems_stft[:, V:]], dim=1)

        Y_v_other = (Y_v_sum - Y_v).reshape(B * V * C, F_bins, T_frames)
        Y_bg_sum = Y_bg_sum_1.expand(B, V, C, F_bins, T_frames).reshape(B * V * C, F_bins, T_frames)
        Y_v_enh = self._apply_stft_cancel(
            self.voice_stft_net,
            Y_v.reshape(B * V * C, F_bins, T_frames),
            Y_v_other,
            Y_bg_sum,
        ).reshape(B, V, C, F_bins, T_frames)
        if self.apply_gate:
            g_v = gate[:, :V].float()
            Y_v_enh = Y_v_enh * g_v.view(B, V, 1, 1, 1)

        Y_f = Y_all[:, V:]
        Y_f_other = Y_f.flip(dims=[1]).reshape(B * 2 * C, F_bins, T_frames)
        Y_v_tot = Y_v_sum.expand(B, 2, C, F_bins, T_frames).reshape(B * 2 * C, F_bins, T_frames)
        Y_f_enh = self._apply_stft_cancel(
            self.bg_stft_net,
            Y_f.reshape(B * 2 * C, F_bins, T_frames),
            Y_f_other,
            Y_v_tot,
        ).reshape(B, 2, C, F_bins, T_frames)

        stems_stft = istft_fn(torch.cat([Y_v_enh, Y_f_enh], dim=1), length=N)
        v_enh = stems_stft[:, :V]
        v_other_w = v_enh.sum(dim=1, keepdim=True) - v_enh
        v_bg_w = (stems_stft[:, V : V + 1] + stems_stft[:, V + 1 : V + 2]).expand(B, V, C, N)
        wave_in = torch.stack([v_enh, v_other_w, v_bg_w], dim=3).reshape(B * V * C, 3, N)
        if self.training and wave_in.requires_grad:
            raw_dw = checkpoint(self.voice_wave_net, wave_in, use_reentrant=False).reshape(B, V, C, N).float()
        else:
            raw_dw = self.voice_wave_net(wave_in).reshape(B, V, C, N).float()
        v_final = _apply_bounded_residual(v_enh, raw_dw)
        if self.apply_gate:
            v_final = v_final * g_v.view(B, V, 1, 1)
        return torch.cat([v_final, stems_stft[:, V:]], dim=1)


def _build_band_groups(
    bands: List[Tuple[int, int]], n_low: int
) -> List[Tuple[int, int, int, int, int, int]]:
    """Group contiguous bands with identical (bw, harm_bw) -> (b0, b1, f0, f1, bw, harm_bw)."""
    groups: List[Tuple[int, int, int, int, int, int]] = []
    b_idx = 0
    while b_idx < len(bands):
        i0, j0 = bands[b_idx]
        bw = j0 - i0
        harm_bw = max(0, min(j0, n_low) - i0)
        b_end = b_idx + 1
        f_end = j0
        if harm_bw == 0 or harm_bw == bw:
            while b_end < len(bands):
                ik, jk = bands[b_end]
                hk = max(0, min(jk, n_low) - ik)
                if (jk - ik) == bw and hk == harm_bw:
                    f_end = jk
                    b_end += 1
                else:
                    break
        groups.append((b_idx, b_end, i0, f_end, bw, harm_bw))
        b_idx = b_end
    return groups


class _GroupedBandEncoder(nn.Module):
    """Batched band encoder across G contiguous variable-width frequency bands."""

    def __init__(self, n_bands_g: int, bw: int, harm_bw: int, feat_per_bin: int, dim: int):
        super().__init__()
        self.n_bands_g = n_bands_g
        self.bw = bw
        self.harm_bw = harm_bw
        self.feat_per_bin = feat_per_bin
        self.dim = dim
        extra = (dim // 2) if harm_bw > 0 else 0
        in_dim = bw * feat_per_bin + extra

        if harm_bw > 0:
            h_in = harm_bw * (dim // 2)
            self.harm_w = nn.Parameter(torch.empty(n_bands_g, h_in, dim // 2))
            self.harm_b = nn.Parameter(torch.zeros(n_bands_g, dim // 2))
            bound_h = 1.0 / math.sqrt(h_in)
            nn.init.uniform_(self.harm_w, -bound_h, bound_h)
            nn.init.uniform_(self.harm_b, -bound_h, bound_h)

        self.ln_w = nn.Parameter(torch.ones(n_bands_g, in_dim))
        self.ln_b = nn.Parameter(torch.zeros(n_bands_g, in_dim))
        self.w = nn.Parameter(torch.empty(n_bands_g, in_dim, dim))
        self.b = nn.Parameter(torch.zeros(n_bands_g, dim))
        bound = 1.0 / math.sqrt(in_dim)
        nn.init.uniform_(self.w, -bound, bound)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(
        self, bin_slice: torch.Tensor, harm_slice: torch.Tensor | None
    ) -> torch.Tensor:
        B, T, _, _ = bin_slice.shape
        G = self.n_bands_g
        sub = bin_slice.reshape(B, T, G, self.bw * self.feat_per_bin)
        comp_dt = (
            torch.get_autocast_dtype("cuda")
            if (sub.is_cuda and torch.is_autocast_enabled("cuda"))
            else self.w.dtype
        )
        if self.harm_bw > 0:
            assert harm_slice is not None
            h_in = harm_slice.reshape(B, T, G, self.harm_bw * (self.dim // 2)).to(comp_dt)
            h_proj = torch.einsum("btgi,gid->btgd", h_in, self.harm_w.to(comp_dt)) + self.harm_b.to(comp_dt)
            sub = torch.cat([sub.to(comp_dt), h_proj], dim=-1)

        sf = sub.float()
        mean = sf.mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt((sf - mean).square().mean(dim=-1, keepdim=True) + 1e-5)
        sub_n = ((sf - mean) * rstd) * self.ln_w.float() + self.ln_b.float()
        out = torch.einsum("btgi,gid->btgd", sub_n.to(comp_dt), self.w.to(comp_dt)) + self.b.to(comp_dt)
        return F.gelu(out)


class _GroupedBandDecoder(nn.Module):
    """Batched voice + fixed-stem complex mask decoder across G contiguous bands."""

    def __init__(self, n_bands_g: int, bw: int, in_channels: int, dim: int):
        super().__init__()
        self.n_bands_g = n_bands_g
        self.bw = bw
        self.in_channels = in_channels
        out_dim = in_channels * bw * 2
        bound = 1.0 / math.sqrt(dim)

        self.w_v = nn.Parameter(torch.empty(n_bands_g, dim, out_dim))
        self.b_v = nn.Parameter(torch.zeros(n_bands_g, out_dim))
        nn.init.uniform_(self.w_v, -bound, bound)
        nn.init.uniform_(self.b_v, -bound, bound)

        self.w_f = nn.Parameter(torch.empty(n_bands_g, 2, dim, out_dim))
        self.b_f = nn.Parameter(torch.zeros(n_bands_g, 2, out_dim))
        nn.init.uniform_(self.w_f, -bound, bound)
        nn.init.uniform_(self.b_f, -bound, bound)

    def forward(self, slot_ctx_g: torch.Tensor, n_voices: int) -> torch.Tensor:
        B, S, T, G, D = slot_ctx_g.shape
        C = self.in_channels
        V = n_voices
        v_h = slot_ctx_g[:, :V].float()
        v_out = (
            torch.einsum("bvtgd,gdo->bvtgo", v_h, self.w_v.float())
            + self.b_v.float().view(1, 1, 1, G, -1)
        ).view(B, V, T, G, C, self.bw, 2).permute(0, 1, 2, 4, 3, 5, 6).reshape(
            B, V, T, C, G * self.bw, 2
        )

        f_h = slot_ctx_g[:, V:].float()
        f_out = (
            torch.einsum("bftgd,gfdo->bftgo", f_h, self.w_f.float())
            + self.b_f.float().view(1, 2, 1, G, -1)
        ).view(B, 2, T, G, C, self.bw, 2).permute(0, 1, 2, 4, 3, 5, 6).reshape(
            B, 2, T, C, G * self.bw, 2
        )
        return torch.cat([v_out, f_out], dim=1)


class HarmonicSlotTF(nn.Module):
    """Multi-stem separator for 2-3 Voices, Music, and Ambience."""

    def __init__(
        self,
        sr: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 256,
        in_channels: int = 2,
        dim: int = 128,
        n_enc_layers: int = 4,
        n_dec_layers: int = 2,
        n_voices: int = 3,
        mode: AblationMode = "purity_queries",
        legacy_slots: bool = False,
        use_inpaint: bool = False,
        use_enhancer: bool = False,
        grad_checkpoint: bool = False,
        arch_version: str = "v2",
    ):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.in_channels = in_channels
        self.n_dec_layers = n_dec_layers
        self.n_voices = n_voices
        self.n_sources = n_voices + 2  # [Voice_0..Voice_{V-1}, Music, Ambience]
        self.mode = mode
        self.legacy_slots = legacy_slots
        self.use_inpaint = use_inpaint
        self.use_enhancer = use_enhancer
        self.grad_checkpoint = grad_checkpoint
        self.arch_version = arch_version

        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        n_freqs = n_fft // 2 + 1
        self.n_freqs = n_freqs
        self.bands = build_variable_bands(n_freqs, sr=sr, n_fft=n_fft)
        self.n_bands = len(self.bands)
        if not legacy_slots:
            self.band_pos = nn.Parameter(torch.randn(1, 1, self.n_bands, dim) * 0.02)

        expected_adv = 2.0 * math.pi * torch.arange(n_freqs, dtype=torch.float32) * (hop_length / n_fft)
        self.register_buffer("expected_adv", expected_adv, persistent=False)

        self.harm_gather = HarmonicStackGather(
            n_freqs=n_freqs, sr=sr, n_fft=n_fft, out_dim=dim // 2
        )
        self.n_low = self.harm_gather.n_low

        feat_per_bin = in_channels * 2 + 3
        if arch_version == "v3":
            self.band_groups = _build_band_groups(self.bands, self.n_low)
            self.band_enc_groups = nn.ModuleList(
                [
                    _GroupedBandEncoder(b1 - b0, bw, harm_bw, feat_per_bin, dim)
                    for (b0, b1, _f0, _f1, bw, harm_bw) in self.band_groups
                ]
            )
            self.band_dec_groups = nn.ModuleList(
                [
                    _GroupedBandDecoder(b1 - b0, bw, in_channels, dim)
                    for (b0, b1, _f0, _f1, bw, _hb) in self.band_groups
                ]
            )
        else:
            self.band_encoders = nn.ModuleList()
            self.harm_proj_per_band = nn.ModuleList()
            # Shared voice decoder + separate Music and Ambience decoders per band
            self.voice_decoders = nn.ModuleList()
            self.fixed_decoders = nn.ModuleList()
            for (i, j) in self.bands:
                bw = j - i
                extra = (dim // 2) if i < self.n_low else 0
                self.band_encoders.append(
                    nn.Sequential(
                        nn.LayerNorm(bw * feat_per_bin + extra),
                        nn.Linear(bw * feat_per_bin + extra, dim),
                        nn.GELU(),
                    )
                )
                self.harm_proj_per_band.append(
                    nn.Linear((min(j, self.n_low) - i) * (dim // 2), dim // 2)
                    if i < self.n_low
                    else nn.Identity()
                )
                self.voice_decoders.append(nn.Linear(dim, in_channels * bw * 2))
                self.fixed_decoders.append(nn.Linear(dim, 2 * in_channels * bw * 2))

        n_heads = 8 if (arch_version == "v3" and dim >= 256) else 4
        use_rope = arch_version == "v3"
        self.encoder = nn.ModuleList(
            [
                DualPathBandBlock(
                    dim,
                    n_heads=n_heads,
                    use_time_attn=not legacy_slots,
                    use_rope=use_rope,
                    use_inter_source=False,
                )
                for _ in range(n_enc_layers)
            ]
        )

        if mode == "plain":
            self.stem_proj = nn.Linear(dim, self.n_sources * dim)
            self.activity_head = nn.Linear(dim, 1)
        else:
            self.slot_module = SlotQueryModule(
                dim,
                n_voices=n_voices,
                n_heads=n_heads,
                use_purity=(mode == "purity_queries"),
                legacy_slots=legacy_slots,
                arch_version=arch_version,
            )
            self.activity_head = nn.Linear(dim, 1)

        self.decoder = nn.ModuleList(
            [
                DualPathBandBlock(
                    dim,
                    n_heads=n_heads,
                    use_time_attn=False,
                    use_rope=use_rope,
                    use_inter_source=(arch_version == "v3"),
                )
                for _ in range(n_dec_layers)
            ]
        )
        self.out_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        if self.use_inpaint:
            self.inpaint_ref = nn.Sequential(
                nn.Conv1d(in_channels, 16, kernel_size=15, padding=7, bias=False),
                nn.SiLU(),
                nn.Conv1d(16, in_channels, kernel_size=15, padding=7, bias=False),
            )
            nn.init.normal_(self.inpaint_ref[-1].weight, mean=0.0, std=1e-3)
        if self.use_enhancer:
            enh_h = 32 if arch_version == "v3" else 16
            self.enhancer = CrossResidueEnhancer(hidden_stft=enh_h, hidden_wave=enh_h, apply_gate=(arch_version == "v3"))

    def _stft(self, wav: torch.Tensor) -> torch.Tensor:
        B, C, N = wav.shape
        X = torch.stft(
            wav.reshape(B * C, N).float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(torch.float32),
            center=True,
            return_complex=True,
        )
        return X.reshape(B, C, self.n_freqs, -1)

    def _istft(self, X: torch.Tensor, length: int) -> torch.Tensor:
        lead = X.shape[:-2]
        Xf = X.reshape(-1, self.n_freqs, X.shape[-1])
        wav = torch.istft(
            Xf,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(Xf.real.dtype),
            center=True,
            length=length,
        )
        return wav.reshape(*lead, length)

    def _compute_physics(
        self, X_btf: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mag_ch = X_btf.abs()
        mag = mag_ch.mean(dim=-1)
        phase = torch.angle(X_btf[..., 0])
        raw_dphi = phase[:, 1:] - phase[:, :-1] - self.expected_adv.view(1, 1, -1)
        if_dev = torch.atan2(torch.sin(raw_dphi), torch.cos(raw_dphi)) / math.pi
        if_dev = F.pad(if_dev, (0, 0, 1, 0))
        if not self.legacy_slots:
            if_dev = if_dev * torch.tanh(mag * 10.0)

        if X_btf.shape[-1] >= 2:
            ild = torch.log1p(mag_ch[..., 0]) - torch.log1p(mag_ch[..., 1])
            ipd_cos = torch.cos(torch.angle(X_btf[..., 0]) - torch.angle(X_btf[..., 1]))
        else:
            ild = torch.zeros_like(mag)
            ipd_cos = torch.ones_like(mag)
        return mag, if_dev, ild, ipd_cos

    def forward(
        self, mix: torch.Tensor, prev_slots: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor | None]:
        B, C, N = mix.shape
        X = self._stft(mix)
        X_btf = X.permute(0, 3, 2, 1)
        T = X_btf.shape[1]

        mag, if_dev, ild, ipd_cos = self._compute_physics(X_btf)
        harm_low = self.harm_gather(mag, if_dev, ild, ipd_cos)

        # Log-compress raw STFT real/imag so large unnormalized bin magnitudes don't drown physics features
        abs_x = X_btf.abs()
        comp_scale = torch.log1p(abs_x) / (abs_x + 1e-6)
        re_im = torch.cat([X_btf.real * comp_scale, X_btf.imag * comp_scale], dim=-1)
        phys = torch.stack([if_dev, ild, ipd_cos], dim=-1)
        bin_feats = torch.cat([re_im, phys], dim=-1)

        if self.arch_version == "v3":
            band_tokens = [
                enc(
                    bin_feats[:, :, f0:f1, :],
                    harm_low[:, :, f0 : f0 + (b1 - b0) * harm_bw, :] if harm_bw > 0 else None,
                )
                for enc, (b0, b1, f0, f1, _bw, harm_bw) in zip(self.band_enc_groups, self.band_groups)
            ]
            z = torch.cat(band_tokens, dim=2)
        else:
            band_tokens = []
            for b_idx, (i, j) in enumerate(self.bands):
                sub = bin_feats[:, :, i:j, :].reshape(B, T, -1)
                if i < self.n_low:
                    hi_h = min(j, self.n_low)
                    h_flat = self.harm_proj_per_band[b_idx](
                        harm_low[:, :, i:hi_h, :].reshape(B, T, -1)
                    )
                    sub = torch.cat([sub, h_flat], dim=-1)
                band_tokens.append(self.band_encoders[b_idx](sub))
            z = torch.stack(band_tokens, dim=2)

        if not self.training:
            del bin_feats, harm_low, mag, if_dev, ild, ipd_cos, abs_x, comp_scale, re_im, phys, X_btf, band_tokens

        if not self.legacy_slots:
            z = z + self.band_pos.to(dtype=z.dtype)
        for blk in self.encoder:
            if self.training and self.grad_checkpoint and z.requires_grad:
                z = checkpoint(blk, z, use_reentrant=False)
            else:
                z = blk(z)

        S = self.n_sources
        V = self.n_voices
        if self.mode == "plain":
            slot_ctx = self.stem_proj(z).view(B, T, self.n_bands, S, -1).permute(0, 3, 1, 2, 4)
            slots = None
            class_logits, solo_spk_logits = None, None
        else:
            slot_ctx, slots, class_logits, solo_spk_logits = self.slot_module(z, prev_slots=prev_slots)
        if not self.training:
            del z

        if len(self.decoder) > 0:
            dec_in = slot_ctx.reshape(B * S, T, self.n_bands, slot_ctx.shape[-1])
            for blk in self.decoder:
                if self.training and self.grad_checkpoint and dec_in.requires_grad:
                    dec_in = checkpoint(blk, dec_in, S, use_reentrant=False)
                else:
                    dec_in = blk(dec_in, S)
            slot_ctx = dec_in.reshape(B, S, T, self.n_bands, slot_ctx.shape[-1])
            if not self.training:
                del dec_in

        if not self.training:
            t_chunk = 128
            for s in range(S):
                for t0 in range(0, T, t_chunk):
                    t1 = min(T, t0 + t_chunk)
                    sc = slot_ctx[:, s : s + 1, t0:t1]
                    out_sc = sc + self.out_mlp(sc)
                    if s == 0 and t0 == 0 and out_sc.dtype != slot_ctx.dtype:
                        slot_ctx = slot_ctx.to(dtype=out_sc.dtype)
                    slot_ctx[:, s : s + 1, t0:t1] = out_sc
                    del out_sc
        else:
            slot_ctx = slot_ctx + self.out_mlp(slot_ctx)

        if self.mode == "plain" or not self.legacy_slots:
            activity_logits = self.activity_head(slot_ctx.mean(dim=(2, 3))).squeeze(-1)
        else:
            assert slots is not None
            activity_logits = self.activity_head(slots).squeeze(-1)

        if self.arch_version == "v3":
            mask_groups = [
                dec(slot_ctx[:, :, :, b0:b1, :], V)
                for dec, (b0, b1, _f0, _f1, _bw, _hb) in zip(self.band_dec_groups, self.band_groups)
            ]
            if not self.training:
                del slot_ctx
            masks_ri = torch.cat(mask_groups, dim=4).float()
            if not self.training:
                del mask_groups
            masks_ri = 2.0 * torch.tanh(masks_ri / 2.0)
        else:
            # Decode symmetric voice masks via shared voice decoder; Music/Ambience via fixed decoder
            mask_bands = []
            for b_idx, (i, j) in enumerate(self.bands):
                bw = j - i
                v_out = self.voice_decoders[b_idx](slot_ctx[:, :V, :, b_idx, :]).view(B, V, T, C, bw, 2)

                f_dec = self.fixed_decoders[b_idx]
                W_f = f_dec.weight.view(2, C * bw * 2, -1)
                b_f = f_dec.bias.view(2, C * bw * 2)
                f_h = slot_ctx[:, V:, :, b_idx, :]
                f_out = (torch.einsum("bstd,fod->bsto", f_h, W_f) + b_f.view(1, 2, 1, -1)).view(B, 2, T, C, bw, 2)

                mask_bands.append(torch.cat([v_out, f_out], dim=1))
            if not self.training:
                del slot_ctx
            masks_ri = torch.cat(mask_bands, dim=4).float()  # [B, S, T, C, F, 2]
            if not self.training:
                del mask_bands

        masks_c = torch.complex(masks_ri[..., 0], masks_ri[..., 1]).permute(0, 1, 3, 4, 2)
        if not self.training:
            del masks_ri

        # Activity gate for voice slots; Music & Ambience stay active (=1.0)
        gate = torch.ones(B, S, device=mix.device, dtype=torch.float32)
        act_for_gate = (
            activity_logits[:, :V].detach()
            if (self.training and not self.legacy_slots)
            else activity_logits[:, :V]
        )
        gate[:, :V] = torch.sigmoid(act_for_gate.float())
        est_stft = (masks_c * gate.view(B, S, 1, 1, 1)) * X.unsqueeze(1)
        if not self.training:
            del masks_c

        if self.arch_version != "v3":
            # v2 mixture consistency residual dump (omitted in v3 to prevent inter-speaker smearing)
            mag_sq = est_stft.real.square() + est_stft.imag.square()
            stem_energy = (mag_sq.square() + 1e-12) * gate.view(B, S, 1, 1, 1)
            weights = stem_energy / stem_energy.sum(dim=1, keepdim=True).clamp_min(1e-12)
            residual = X - est_stft.sum(dim=1)
            est_stft = est_stft + weights * residual.unsqueeze(1)

        stems = self._istft(est_stft, length=N)
        if self.use_inpaint and solo_spk_logits is not None:
            p_imp = F.interpolate(
                (1.0 - torch.sigmoid(solo_spk_logits.float())).mean(dim=2).permute(0, 2, 1),
                size=N,
                mode="linear",
                align_corners=False,
            ).unsqueeze(1)
            v_wav = stems[:, :V]
            raw_dv = self.inpaint_ref(v_wav.reshape(B * V, C, N)).reshape(B, V, C, N)
            v_imp = _apply_bounded_residual(v_wav, raw_dv, mask=p_imp)
            stems = torch.cat([v_imp, stems[:, V:]], dim=1)
        if self.use_enhancer:
            stems = self.enhancer(stems, gate, self._stft, self._istft)
        return {
            "stems": stems,
            "slots": slots,
            "activity_logits": activity_logits,
            "class_logits": class_logits,
            "solo_spk_logits": solo_spk_logits,
        }
