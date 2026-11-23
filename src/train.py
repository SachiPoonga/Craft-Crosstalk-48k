"""Colab-ready training & 3-way ablation script for HarmonicSlotTF.

Supports:
- On-the-fly mixing from clean Speech (speaker-grouped), Music, Ambience, and optional RIR dirs.
- Leak-proof disjoint speaker/file train-val split.
- Auto GPU precision: bf16 on L4/A100/G4, fp16 + GradScaler on T4.
- Separate held-out evaluation for 2-speaker vs 3-speaker SI-SDR / SI-SDRi,
  Music/Ambience SI-SDR, speaker-count accuracy, and silent-slot leakage.
- Atomic checkpoint saving & clean --resume support for Google Drive.
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loss import HybridSeparationLoss
from model import HarmonicSlotTF

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def _load_audio_slice(
    path: Path, target_sr: int, seg_samples: int, rng: random.Random, random_crop: bool = True
) -> torch.Tensor:
    """Load an audio slice of length ~seg_samples (seeking directly when soundfile is available)."""
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(path))
        src_sr = int(info.samplerate)
        total_frames = int(info.frames)
        need_frames = max(1, int(math.ceil(seg_samples * src_sr / target_sr)))
        with sf.SoundFile(str(path)) as f:
            if total_frames > need_frames:
                start = rng.randint(0, total_frames - need_frames) if random_crop else 0
                f.seek(start)
                data = f.read(need_frames, dtype="float32", always_2d=True)
            else:
                data = f.read(dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T.copy())  # [C, N]
    except Exception:
        import torchaudio  # type: ignore

        wav, src_sr = torchaudio.load(str(path))
        wav = wav.float()
        need_frames = max(1, int(math.ceil(seg_samples * src_sr / target_sr)))
        if wav.shape[-1] > need_frames:
            start = rng.randint(0, wav.shape[-1] - need_frames) if random_crop else 0
            wav = wav[:, start : start + need_frames]

    if src_sr != target_sr:
        try:
            import torchaudio  # type: ignore

            wav = torchaudio.functional.resample(wav, src_sr, target_sr)
        except Exception:
            new_len = max(1, int(round(wav.shape[-1] * target_sr / src_sr)))
            wav = F.interpolate(wav.unsqueeze(0), size=new_len, mode="linear", align_corners=False).squeeze(0)
    return wav


def _fit_channels(
    wav: torch.Tensor,
    target_ch: int,
    rng: random.Random,
    force_mono: bool = False,
    pan: Optional[float] = None,
) -> torch.Tensor:
    c = wav.shape[0]
    if target_ch == 1:
        return wav.mean(dim=0, keepdim=True) if c > 1 else wav
    if force_mono and target_ch == 2:
        mono = wav.mean(dim=0, keepdim=True) if c > 1 else wav[:1]
        return mono.expand(2, -1).contiguous()
    if c == target_ch:
        return wav
    if c == 1 and target_ch == 2:
        # Random subtle stereo pan for mono sources when training stereo
        p = rng.uniform(0.3, 0.7) if pan is None else pan
        return torch.cat([wav * math.sqrt(p), wav * math.sqrt(1.0 - p)], dim=0)
    return wav[:target_ch]


def _fit_length(wav: torch.Tensor, target_len: int, rng: random.Random, loop: bool = False) -> torch.Tensor:
    n = wav.shape[-1]
    if n >= target_len:
        start = rng.randint(0, n - target_len) if n > target_len else 0
        return wav[:, start : start + target_len]
    if loop and n > 0:
        reps = (target_len + n - 1) // n
        return wav.repeat(1, reps)[:, :target_len]
    out = torch.zeros(wav.shape[0], target_len, dtype=wav.dtype)
    pos = rng.randint(0, target_len - n)
    out[:, pos : pos + n] = wav
    return out


def _rms(wav: torch.Tensor) -> torch.Tensor:
    return wav.pow(2).mean().sqrt().clamp_min(1e-8)


def _scale_to_db(wav: torch.Tensor, target_db: float) -> torch.Tensor:
    target_rms = 10.0 ** (target_db / 20.0)
    return wav * (target_rms / _rms(wav))


def _discover_speakers(speech_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for p in sorted(speech_dir.rglob("*")):
        if p.suffix.lower() in AUDIO_EXTS and p.is_file():
            rel = p.relative_to(speech_dir)
            spk = rel.parts[0] if len(rel.parts) > 1 else p.stem
            groups.setdefault(spk, []).append(p)
    return groups


def _discover_files(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in AUDIO_EXTS and p.is_file()]


class DynamicMixDataset(Dataset):
    """On-the-fly 2-to-3 speaker + Music + Ambience mixer with disjoint splits."""

    def __init__(
        self,
        speech_groups: Dict[str, List[Path]],
        music_files: List[Path],
        amb_files: List[Path],
        rir_files: List[Path],
        sr: int,
        channels: int,
        seg_seconds: float,
        n_samples_epoch: int,
        seed: int = 42,
        fixed_n_active: Optional[int] = None,
        p_3spk: float = 0.5,
        hard_lang_prob: float = 0.0,
        full_overlap_prob: float = 0.0,
        force_full_overlap: bool = False,
        stream_prefix_len: int = 0,
        deterministic: bool = False,
        mono_mix_prob: float = 0.0,
    ):
        self.spk_ids = sorted(speech_groups.keys())
        assert len(self.spk_ids) >= 3, f"Need >=3 distinct speakers, got {len(self.spk_ids)}"
        self.speech_groups = speech_groups
        self.music_files = music_files
        self.amb_files = amb_files
        self.rir_files = rir_files
        self.sr = sr
        self.channels = channels
        self.seg_len = int(round(sr * seg_seconds))
        self.n_samples_epoch = n_samples_epoch
        self.seed = seed
        self.fixed_n_active = fixed_n_active
        self.p_3spk = p_3spk
        self.hard_lang_prob = hard_lang_prob
        self.full_overlap_prob = full_overlap_prob
        self.force_full_overlap = force_full_overlap
        self.stream_prefix_len = stream_prefix_len
        self.deterministic = deterministic
        self.mono_mix_prob = mono_mix_prob

        self.lang_groups: Dict[str, List[str]] = {}
        for s in self.spk_ids:
            lang = s.split("_")[0] if "_" in s else "unk"
            self.lang_groups.setdefault(lang, []).append(s)
        self.hard_langs = [k for k, v in self.lang_groups.items() if len(v) >= 3]

    def __len__(self) -> int:
        return self.n_samples_epoch

    def _load_speaker_track(
        self,
        utts: List[Path],
        target_len: int,
        rng: random.Random,
        force_mono: bool = False,
        pan: Optional[float] = None,
    ) -> torch.Tensor:
        """Load target_len audio for a speaker, stitching distinct utterances rather than repeating."""
        if not force_mono and pan is None and self.channels == 2:
            pan = rng.uniform(0.3, 0.7)
        if len(utts) == 1:
            w = _load_audio_slice(utts[0], self.sr, target_len, rng)
            return _fit_length(
                _fit_channels(w, self.channels, rng, force_mono=force_mono, pan=pan),
                target_len, rng, loop=True,
            )

        chunks: List[torch.Tensor] = []
        total = 0
        cand = list(utts)
        rng.shuffle(cand)
        idx = 0
        while total < target_len:
            u = cand[idx % len(cand)]
            idx += 1
            if idx % len(cand) == 0:
                rng.shuffle(cand)
            need = target_len - total
            w = _load_audio_slice(u, self.sr, need, rng)
            w = _fit_channels(w, self.channels, rng, force_mono=force_mono, pan=pan)
            if w.shape[-1] > need:
                start = rng.randint(0, w.shape[-1] - need)
                w = w[:, start : start + need]
            chunks.append(w)
            total += w.shape[-1]
            if total < target_len:
                gap_len = min(target_len - total, rng.randint(int(0.1 * self.sr), int(0.4 * self.sr)))
                if gap_len > 0:
                    chunks.append(torch.zeros(self.channels, gap_len, dtype=w.dtype))
                    total += gap_len
        return torch.cat(chunks, dim=-1)[:, :target_len]

    def _apply_rir(self, wav: torch.Tensor, rir_path: Path, rng: random.Random) -> torch.Tensor:
        # Always read RIR from t=0 so the direct-path impulse peak is preserved
        rir = _load_audio_slice(rir_path, self.sr, int(0.5 * self.sr), rng, random_crop=False).mean(dim=0)
        rir = rir[: int(0.4 * self.sr)]
        norm = rir.pow(2).sum().sqrt().clamp_min(1e-6)
        rir = rir / norm
        n = wav.shape[-1]
        L = n + rir.numel() - 1
        nfft = 1 << (L - 1).bit_length()
        W = torch.fft.rfft(wav, n=nfft)
        R = torch.fft.rfft(rir, n=nfft).unsqueeze(0)
        conv = torch.fft.irfft(W * R, n=nfft)[:, :n]
        return conv

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.fixed_n_active is not None or self.deterministic:
            rng = random.Random(self.seed + idx)
        else:
            # Use PyTorch RNG (uniquely seeded per DataLoader worker & epoch) to avoid duplicate mixes
            seed_val = int(torch.randint(0, 2**31 - 1, (1,)).item())
            rng = random.Random(seed_val)
        n_active = (
            self.fixed_n_active
            if self.fixed_n_active is not None
            else (3 if rng.random() < self.p_3spk else 2)
        )
        if (
            self.fixed_n_active is None
            and not self.deterministic
            and self.hard_langs
            and rng.random() < self.hard_lang_prob
        ):
            lang = rng.choice(self.hard_langs)
            chosen_spks = rng.sample(self.lang_groups[lang], n_active)
        else:
            chosen_spks = rng.sample(self.spk_ids, n_active)

        stems = torch.zeros(5, self.channels, self.seg_len, dtype=torch.float32)
        base_speech_db = rng.uniform(-22.0, -16.0)
        is_mono_mix = (self.channels == 2) and (self.mono_mix_prob > 0.0) and (rng.random() < self.mono_mix_prob)
        is_full_ov = self.force_full_overlap or (
            self.fixed_n_active is None and not self.deterministic and rng.random() < self.full_overlap_prob
        )

        for v in range(n_active):
            utts = self.speech_groups[chosen_spks[v]]
            min_keep = int(0.65 * self.seg_len)
            if self.stream_prefix_len > 0:
                keep = self.seg_len
            else:
                keep = self.seg_len if is_full_ov else rng.randint(min_keep, self.seg_len)
            spk_pan = None if is_mono_mix else rng.uniform(0.3, 0.7)
            w = self._load_speaker_track(utts, keep, rng, force_mono=is_mono_mix, pan=spk_pan)
            if keep < self.seg_len:
                w = _fit_length(w, self.seg_len, rng, loop=False)
            if self.rir_files and rng.random() < 0.5:
                w = self._apply_rir(w, rng.choice(self.rir_files), rng)
            if _rms(w) > 1e-6:
                w = _scale_to_db(w, base_speech_db + rng.uniform(-3.0, 3.0))
            stems[v] = w

        # Music stem (15% silent during training so model learns clean speech-only mixes too)
        if self.music_files and (self.fixed_n_active is not None or self.deterministic or rng.random() > 0.15):
            mw = _load_audio_slice(rng.choice(self.music_files), self.sr, self.seg_len, rng)
            mw = _fit_channels(mw, self.channels, rng, force_mono=is_mono_mix)
            mw = _fit_length(mw, self.seg_len, rng, loop=True)
            if _rms(mw) > 1e-6:
                mw = _scale_to_db(mw, base_speech_db + rng.uniform(-15.0, 3.0))
            stems[3] = mw

        # Ambience stem
        if self.amb_files and (self.fixed_n_active is not None or self.deterministic or rng.random() > 0.15):
            aw = _load_audio_slice(rng.choice(self.amb_files), self.sr, self.seg_len, rng)
            aw = _fit_channels(aw, self.channels, rng, force_mono=is_mono_mix)
            aw = _fit_length(aw, self.seg_len, rng, loop=True)
            if _rms(aw) > 1e-6:
                aw = _scale_to_db(aw, base_speech_db + rng.uniform(-20.0, -3.0))
            stems[4] = aw

        mix = stems.sum(dim=0)
        peak = mix.abs().max()
        if peak > 0.95:
            scale = 0.95 / peak
            stems = stems * scale
            mix = mix * scale
        return mix, stems, n_active


class SyntheticMixDataset(Dataset):
    """Fast synthetic speech-like harmonic + music + noise dataset for local dry runs."""

    def __init__(self, sr: int, channels: int, seg_seconds: float, length: int, fixed_n_active: Optional[int] = None):
        self.sr = sr
        self.channels = channels
        self.seg_len = int(round(sr * seg_seconds))
        self.length = length
        self.fixed_n_active = fixed_n_active

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        g = torch.Generator().manual_seed(idx + 1234)
        n_active = self.fixed_n_active if self.fixed_n_active is not None else (2 if idx % 2 == 0 else 3)
        t = torch.arange(self.seg_len, dtype=torch.float32) / self.sr
        stems = torch.zeros(5, self.channels, self.seg_len, dtype=torch.float32)

        for v in range(n_active):
            f0 = float(torch.empty(1).uniform_(95, 280, generator=g))
            sig = sum((0.6 ** h) * torch.sin(2 * math.pi * f0 * (h + 1) * t) for h in range(4))
            env = 0.5 + 0.5 * torch.sin(2 * math.pi * (2.5 + v) * t)
            sig = (sig * env) * 0.08
            stems[v] = sig.unsqueeze(0).expand(self.channels, -1)

        # Synthetic music chord + ambience noise
        chord = sum(torch.sin(2 * math.pi * f * t) for f in (220.0, 277.18, 329.63)) * 0.04
        stems[3] = chord.unsqueeze(0).expand(self.channels, -1)
        stems[4] = torch.randn(self.channels, self.seg_len, generator=g) * 0.02
        return stems.sum(dim=0), stems, n_active


def sisdr(est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-Invariant SDR (dB) along last dim: est, target [..., N]."""
    s = target - target.mean(dim=-1, keepdim=True)
    sh = est - est.mean(dim=-1, keepdim=True)
    s_pow = s.pow(2).sum(dim=-1, keepdim=True)
    alpha = (sh * s).sum(dim=-1, keepdim=True) / (s_pow + 1e-8)
    e_tgt = alpha * s
    e_res = sh - e_tgt
    return 10.0 * torch.log10((e_tgt.pow(2).sum(dim=-1) + 1e-8) / (e_res.pow(2).sum(dim=-1) + 1e-8))


def _compute_batch_sir(
    preds_v: torch.Tensor,
    targets_v: torch.Tensor,
    best_perm: Tuple[int, ...],
    n_act: int,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor,
) -> List[float]:
    """Cross-speaker SIR (dB) on interferer-dominant STFT bins for one sample [V, C, N]."""
    V, C, N = targets_v.shape
    if n_act <= 1:
        return []
    win = window.to(targets_v.device)
    P = torch.stft(preds_v.reshape(V * C, N).float(), n_fft, hop_length, n_fft, win, center=True, return_complex=True).abs()
    T = torch.stft(targets_v[:n_act].reshape(n_act * C, N).float(), n_fft, hop_length, n_fft, win, center=True, return_complex=True).abs()
    _, F_b, T_f = P.shape
    P = P.reshape(V, C, F_b, T_f)
    T = T.reshape(n_act, C, F_b, T_f)
    t_mean = T.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
    sirs: List[float] = []
    for i in range(n_act):
        est_i = P[best_perm[i]]
        for j in range(n_act):
            if i == j:
                continue
            dom = (T[j] > 0.10 * t_mean[j]) & (T[i] < 0.15 * T[j])
            if bool(dom.any()):
                sig_p = float(T[j][dom].pow(2).mean().item())
                leak_p = float(est_i[dom].pow(2).mean().item())
                sirs.append(10.0 * math.log10((sig_p + 1e-12) / (leak_p + 1e-12)))
    return sirs


def _prepare_carry_slots(
    model: HarmonicSlotTF,
    raw_slots: Optional[torch.Tensor],
    perms: List[Tuple[int, ...]],
    tgt_wavs: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Detach voice slots and reset inactive slots to voice_queries (matching ComfyUI inference)."""
    if raw_slots is None:
        return None
    V = model.n_voices
    carry = raw_slots[:, :V].detach().clone()
    has_vq = hasattr(model, "slot_module") and hasattr(model.slot_module, "voice_queries")
    if not has_vq:
        return carry
    rms = tgt_wavs[:, :V].float().pow(2).mean(dim=(2, 3)).sqrt() > 1e-4  # [B, V]
    vq = model.slot_module.voice_queries.to(device=carry.device, dtype=carry.dtype)
    for b in range(carry.shape[0]):
        for idx_p, idx_t in enumerate(perms[b]):
            if not bool(rms[b, idx_t]):
                carry[b, idx_p] = vq[0, idx_p]
    return carry


@torch.no_grad()
def evaluate(
    model: HarmonicSlotTF,
    loader_2spk: DataLoader,
    loader_3spk: DataLoader,
    device: str,
    amp_dtype: torch.dtype,
    amp_enabled: bool = False,
    loader_ov2: Optional[DataLoader] = None,
    loader_ov3: Optional[DataLoader] = None,
    loader_stream: Optional[DataLoader] = None,
    stream_chunk_seconds: float = 6.0,
    stream_overlap_seconds: float = 1.5,
    use_prev_slots: bool = True,
) -> Dict[str, float]:
    model.eval()
    V = model.n_voices
    metrics: Dict[str, List[float]] = {
        "sisdr_2spk": [],
        "sisdri_2spk": [],
        "sisdr_3spk": [],
        "sisdri_3spk": [],
        "sisdr_ov2": [],
        "sisdr_ov3": [],
        "sir_bleed": [],
        "swap_rate": [],
        "gap_swap_rate": [],
        "sisdr_music": [],
        "sisdr_amb": [],
        "count_acc": [],
        "unused_slot_dbfs": [],
        "voice_scale": [],
    }
    sir_win = torch.hann_window(model.n_fft, device=device)

    eval_sets: List[Tuple[str, int, DataLoader, bool]] = [
        ("2spk", 2, loader_2spk, False),
        ("3spk", 3, loader_3spk, False),
    ]
    if loader_ov2 is not None:
        eval_sets.append(("ov2", 2, loader_ov2, True))
    if loader_ov3 is not None:
        eval_sets.append(("ov3", 3, loader_ov3, True))

    for tag, n_act, loader, compute_sir in eval_sets:
        perms = list(itertools.permutations(range(V), n_act))
        for mix, targets, _ in loader:
            mix = mix.to(device)
            targets = targets.to(device)
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                out = model(mix)
            preds = out["stems"].float()
            act_probs = torch.sigmoid(out["activity_logits"][:, :V].float())
            if tag in ("2spk", "3spk"):
                pred_cnt = (act_probs > 0.5).sum(dim=1)
                metrics["count_acc"].extend((pred_cnt == n_act).float().cpu().tolist())

            B = mix.shape[0]
            for b in range(B):
                best_sdr = -1e9
                best_perm = perms[0]
                for perm in perms:
                    sdr_vals = [sisdr(preds[b, perm[j]], targets[b, j]).mean().item() for j in range(n_act)]
                    mean_s = sum(sdr_vals) / n_act
                    if mean_s > best_sdr:
                        best_sdr = mean_s
                        best_perm = perm

                metrics[f"sisdr_{tag}"].append(best_sdr)
                if tag in ("2spk", "3spk"):
                    base_sdrs = [sisdr(mix[b], targets[b, j]).mean().item() for j in range(n_act)]
                    metrics[f"sisdri_{tag}"].append(best_sdr - sum(base_sdrs) / n_act)
                    for j in range(n_act):
                        p_rms = float(preds[b, best_perm[j]].pow(2).mean().sqrt())
                        t_rms = float(targets[b, j].pow(2).mean().sqrt().clamp_min(1e-8))
                        metrics["voice_scale"].append(p_rms / t_rms)
                    if n_act == 2:
                        unused = next(idx for idx in range(V) if idx not in best_perm)
                        rms_unused = float(preds[b, unused].pow(2).mean().sqrt().clamp_min(1e-10))
                        metrics["unused_slot_dbfs"].append(20.0 * math.log10(rms_unused))
                    if targets[b, 3].abs().max() > 1e-4:
                        metrics["sisdr_music"].append(sisdr(preds[b, 3], targets[b, 3]).mean().item())
                    if targets[b, 4].abs().max() > 1e-4:
                        metrics["sisdr_amb"].append(sisdr(preds[b, 4], targets[b, 4]).mean().item())
                if compute_sir:
                    metrics["sir_bleed"].extend(
                        _compute_batch_sir(preds[b, :V], targets[b, :V], best_perm, n_act, model.n_fft, model.hop_length, sir_win)
                    )

    if loader_stream is not None:
        c_len = int(round(model.sr * stream_chunk_seconds))
        ov_len = int(round(model.sr * stream_overlap_seconds))
        for mix, targets, n_act_t in loader_stream:
            if mix.shape[-1] < c_len + max(1, c_len - ov_len):
                continue
            mix = mix.to(device)
            targets = targets.to(device)
            mix1 = mix[..., :c_len]
            tgt1 = targets[..., :c_len]
            start2 = c_len - ov_len
            mix2 = mix[..., start2 : start2 + c_len]
            tgt2 = targets[..., start2 : start2 + c_len]
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                out1 = model(mix1)
            p1 = out1["stems"].float()
            perm1_list: List[Tuple[int, ...]] = []
            full_perms1: List[Tuple[int, ...]] = []
            for b in range(mix.shape[0]):
                n_act = int(n_act_t[b])
                perms_act = list(itertools.permutations(range(V), n_act))
                perm1 = max(perms_act, key=lambda prm: sum(sisdr(p1[b, prm[j]], tgt1[b, j]).mean().item() for j in range(n_act)))
                perm1_list.append(perm1)
                s2t = [-1] * V
                for j, idx_p in enumerate(perm1):
                    s2t[idx_p] = j
                unused_t = [t for t in range(n_act, V)]
                ui = 0
                for idx_p in range(V):
                    if s2t[idx_p] == -1:
                        s2t[idx_p] = unused_t[ui]
                        ui += 1
                full_perms1.append(tuple(s2t))
            carry = _prepare_carry_slots(model, out1.get("slots"), full_perms1, tgt1) if use_prev_slots else None
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                out2 = model(mix2, prev_slots=carry)
            p2 = out2["stems"].float()
            for b in range(mix.shape[0]):
                n_act = int(n_act_t[b])
                perms_act = list(itertools.permutations(range(V), n_act))
                perm2 = max(perms_act, key=lambda prm: sum(sisdr(p2[b, prm[j]], tgt2[b, j]).mean().item() for j in range(n_act)))
                metrics["swap_rate"].append(1.0 if perm1_list[b] != perm2 else 0.0)

            gap_len = int(round(model.sr * 2.0))
            start2_gap = c_len + gap_len
            if mix.shape[-1] >= start2_gap + c_len:
                mix2_g = mix[..., start2_gap : start2_gap + c_len]
                tgt2_g = targets[..., start2_gap : start2_gap + c_len]
                with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                    out2_g = model(mix2_g, prev_slots=carry)
                p2_g = out2_g["stems"].float()
                for b in range(mix.shape[0]):
                    n_act = int(n_act_t[b])
                    perms_act = list(itertools.permutations(range(V), n_act))
                    perm2_g = max(perms_act, key=lambda prm: sum(sisdr(p2_g[b, prm[j]], tgt2_g[b, j]).mean().item() for j in range(n_act)))
                    metrics["gap_swap_rate"].append(1.0 if perm1_list[b] != perm2_g else 0.0)

    model.train()
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in metrics.items()}


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_safetensors(
    path: Path, state_dict: dict[str, torch.Tensor], meta: dict, cast_bf16: bool = False
) -> None:
    """Save weights as .safetensors with stringified metadata if safetensors is installed."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tensors = {
        k: (
            v.detach().cpu().to(torch.bfloat16).contiguous()
            if (cast_bf16 and v.is_floating_point())
            else v.detach().cpu().contiguous()
        )
        for k, v in state_dict.items()
    }
    str_meta = {k: str(v) for k, v in meta.items()}
    save_file(tensors, str(tmp), metadata=str_meta)
    os.replace(tmp, path)


class ModelEMA:
    """Floating-point EMA shadow weights (stored on GPU in fp32 for fast in-place updates)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().float().clone()
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        d = min(self.decay, (1.0 + float(self.num_updates)) / (10.0 + float(self.num_updates)))
        one_m_d = 1.0 - d
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].lerp_(v.detach().float(), one_m_d)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict, device: str) -> None:
        self.num_updates = int(state.get("num_updates", 0))
        loaded = state.get("shadow", {})
        for k in self.shadow:
            if k in loaded:
                self.shadow[k] = loaded[k].to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def export_state_dict(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            if k in self.shadow:
                out[k] = self.shadow[k].to(device=v.device, dtype=v.dtype)
            else:
                out[k] = v.detach().clone()
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train & ablate HarmonicSlotTF")
    p.add_argument("--arch", choices=["v2", "v3"], default="v2")
    p.add_argument("--mode", choices=["plain", "queries", "purity_queries"], default="purity_queries")
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--channels", type=int, default=2)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=256)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--n-enc-layers", type=int, default=4)
    p.add_argument("--n-dec-layers", type=int, default=2, help="Number of post-slot DualPath decoder layers per stem")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16", "auto"], default="fp32")
    p.add_argument("--save-bf16", action="store_true", help="Export safetensors weights in bfloat16 (keeps last.pt fp32)")
    p.add_argument("--use-inpaint", action="store_true", default=True)
    p.add_argument("--no-inpaint", dest="use_inpaint", action="store_false")
    p.add_argument("--use-enhancer", action="store_true", default=True)
    p.add_argument("--no-enhancer", dest="use_enhancer", action="store_false")
    p.add_argument("--seg-seconds", type=float, default=3.0)
    p.add_argument("--seg-seconds-max", type=float, default=0.0, help="If > seg-seconds, randomly slice batches in [seg-seconds, seg-seconds-max]")
    p.add_argument("--val-seg-seconds", type=float, default=0.0, help="Validation segment length (0 = use --seg-seconds)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps per optimizer update")
    p.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing on DualPath blocks")
    p.add_argument("--p-3spk", type=float, default=0.5, help="Probability of 3-speaker mixes during training")
    p.add_argument("--hard-lang-prob", type=float, default=0.0, help="Probability of sampling same-language speaker mixes")
    p.add_argument("--full-overlap-prob", type=float, default=0.0, help="Probability of 100%-overlap speech clips during training")
    p.add_argument("--mono-mix-prob", type=float, default=0.5, help="Probability of strictly dual-mono (L==R) mixes during stereo training/eval")
    p.add_argument("--fast-pit", action="store_true", help="Match voice slots via fast L1+SISDR matrix before MRSTFT")
    p.add_argument("--activity-weight", type=float, default=0.1)
    p.add_argument("--mix-cons-weight", type=float, default=0.0)
    p.add_argument("--bleed-weight", type=float, default=1.0)
    p.add_argument("--tf-bleed-weight", type=float, default=0.25)
    p.add_argument("--cross-bleed-weight", type=float, default=0.0)
    p.add_argument("--ortho-weight", type=float, default=0.2)
    p.add_argument("--streaming-prob", type=float, default=0.0, help="Probability of paired prev_slots streaming step")
    p.add_argument("--stream-chunk-seconds", type=float, default=6.0)
    p.add_argument("--stream-overlap-seconds", type=float, default=1.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ema-decay", type=float, default=0.999, help="EMA weight decay (0 disables EMA)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--finetune", action="store_true", help="Resume model weights only with fresh Stage-2 optimizer, best_sdr, and LR schedule")
    p.add_argument("--reset-refiners", action="store_true", help="Re-init inpaint_ref & enhancer and reset optimizer/best_sdr on resume")
    p.add_argument("--speech-dir", type=Path, default=None)
    p.add_argument("--music-dir", type=Path, default=None)
    p.add_argument("--amb-dir", type=Path, default=None)
    p.add_argument("--rir-dir", type=Path, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--synthetic-smoke", action="store_true", help="Run fast synthetic smoke test")
    return p.parse_args()


def _reset_refiners(model: HarmonicSlotTF) -> None:
    """Re-initialize post-Wiener refiner modules cleanly to recover from earlier amplitude collapse."""
    heads: List[torch.nn.Module] = []
    if getattr(model, "use_inpaint", False) and hasattr(model, "inpaint_ref"):
        for mod in model.inpaint_ref.modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
        heads.append(model.inpaint_ref[-1])
    if getattr(model, "use_enhancer", False) and hasattr(model, "enhancer"):
        for mod in model.enhancer.modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
        heads.extend([model.enhancer.voice_stft_net[-1], model.enhancer.bg_stft_net[-1], model.enhancer.voice_wave_net[-1]])
    for h in heads:
        torch.nn.init.normal_(h.weight, mean=0.0, std=1e-3)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.precision == "fp32" or device != "cuda":
        amp_enabled = False
        amp_dtype = torch.float32
        use_scaler = False
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    elif args.precision == "bf16":
        amp_enabled = True
        amp_dtype = torch.bfloat16
        use_scaler = False
    elif args.precision == "fp16":
        amp_enabled = True
        amp_dtype = torch.float16
        use_scaler = True
    else:
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_enabled = True
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        use_scaler = not use_bf16
    print(f"Arch={args.arch} | Mode={args.mode} | Device={device} | Precision={args.precision} (amp={amp_enabled}, dtype={amp_dtype}) | Scaler={use_scaler}")

    stream_gap_max_sec = 3.0
    stream_total_sec = (2.0 * args.stream_chunk_seconds + stream_gap_max_sec) if args.streaming_prob > 0.0 else 0.0
    train_seg_max = max(args.seg_seconds, args.seg_seconds_max, stream_total_sec)
    val_seg = args.val_seg_seconds if args.val_seg_seconds > 0.0 else args.seg_seconds
    stream_eval_sec = 2.0 * args.stream_chunk_seconds + 2.0
    if args.synthetic_smoke:
        train_ds = SyntheticMixDataset(args.sr, args.channels, train_seg_max, length=64)
        val_2_ds = SyntheticMixDataset(args.sr, args.channels, val_seg, length=4, fixed_n_active=2)
        val_3_ds = SyntheticMixDataset(args.sr, args.channels, val_seg, length=4, fixed_n_active=3)
        val_ov2_ds = SyntheticMixDataset(args.sr, args.channels, val_seg, length=4, fixed_n_active=2)
        val_ov3_ds = SyntheticMixDataset(args.sr, args.channels, val_seg, length=4, fixed_n_active=3)
        val_stream_ds = SyntheticMixDataset(args.sr, args.channels, stream_eval_sec, length=4) if args.arch == "v3" else None
    else:
        assert args.speech_dir and args.music_dir and args.amb_dir, "Provide --speech-dir, --music-dir, --amb-dir (or --synthetic-smoke)"
        spk_groups = _discover_speakers(args.speech_dir)
        spk_ids = sorted(spk_groups.keys())
        assert len(spk_ids) >= 6, (
            f"Need >=6 distinct speaker folders/files for 3+ train and 3+ val speakers, got {len(spk_ids)}"
        )
        rng = random.Random(42)
        rng.shuffle(spk_ids)
        n_val_spk = min(len(spk_ids) - 3, max(3, int(len(spk_ids) * 0.1)))
        val_spks = {s: spk_groups[s] for s in spk_ids[:n_val_spk]}
        trn_spks = {s: spk_groups[s] for s in spk_ids[n_val_spk:]}

        music_all = _discover_files(args.music_dir)
        amb_all = _discover_files(args.amb_dir)
        rng.shuffle(music_all)
        rng.shuffle(amb_all)
        n_vm = max(1, int(len(music_all) * 0.1))
        n_va = max(1, int(len(amb_all) * 0.1))
        rir_all = _discover_files(args.rir_dir) if args.rir_dir else []
        if len(rir_all) >= 2:
            rng.shuffle(rir_all)
            n_vr = max(1, int(len(rir_all) * 0.1))
            trn_rirs, val_rirs = rir_all[n_vr:], rir_all[:n_vr]
        else:
            trn_rirs, val_rirs = rir_all, rir_all
        print(
            f"Train spks={len(trn_spks)}, Val spks={len(val_spks)}, "
            f"Music={len(music_all)}, Amb={len(amb_all)}, RIR={len(trn_rirs)}+{len(val_rirs)}"
        )

        stream_prefix = (
            int(round(args.sr * max(0.0, args.stream_chunk_seconds - args.stream_overlap_seconds)))
            if args.streaming_prob > 0.0
            else 0
        )
        train_ds = DynamicMixDataset(
            trn_spks, music_all[n_vm:], amb_all[n_va:], trn_rirs,
            args.sr, args.channels, train_seg_max, n_samples_epoch=100000, seed=42,
            p_3spk=args.p_3spk, hard_lang_prob=args.hard_lang_prob, full_overlap_prob=args.full_overlap_prob,
            stream_prefix_len=stream_prefix, mono_mix_prob=args.mono_mix_prob,
        )
        val_2_ds = DynamicMixDataset(val_spks, music_all[:n_vm], amb_all[:n_va], val_rirs, args.sr, args.channels, val_seg, n_samples_epoch=24, seed=100, fixed_n_active=2, mono_mix_prob=args.mono_mix_prob)
        val_3_ds = DynamicMixDataset(val_spks, music_all[:n_vm], amb_all[:n_va], val_rirs, args.sr, args.channels, val_seg, n_samples_epoch=24, seed=200, fixed_n_active=3, mono_mix_prob=args.mono_mix_prob)
        val_ov2_ds = DynamicMixDataset(val_spks, music_all[:n_vm], amb_all[:n_va], val_rirs, args.sr, args.channels, val_seg, n_samples_epoch=24, seed=300, fixed_n_active=2, force_full_overlap=True, mono_mix_prob=args.mono_mix_prob)
        val_ov3_ds = DynamicMixDataset(val_spks, music_all[:n_vm], amb_all[:n_va], val_rirs, args.sr, args.channels, val_seg, n_samples_epoch=24, seed=400, fixed_n_active=3, force_full_overlap=True, mono_mix_prob=args.mono_mix_prob)
        val_stream_ds = (
            DynamicMixDataset(
                val_spks, music_all[:n_vm], amb_all[:n_va], val_rirs,
                args.sr, args.channels, stream_eval_sec, n_samples_epoch=24, seed=500,
                p_3spk=0.5, force_full_overlap=True, deterministic=True, mono_mix_prob=args.mono_mix_prob,
            )
            if args.arch == "v3"
            else None
        )

    nw = 0 if args.synthetic_smoke else args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=(device == "cuda"))
    val_2_loader = DataLoader(val_2_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw)
    val_3_loader = DataLoader(val_3_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw)
    val_ov2_loader = DataLoader(val_ov2_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw)
    val_ov3_loader = DataLoader(val_ov3_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw)
    val_stream_loader = DataLoader(val_stream_ds, batch_size=min(2, args.batch_size), shuffle=False, num_workers=nw) if val_stream_ds is not None else None

    use_inpaint = bool(args.use_inpaint and args.mode == "purity_queries")
    use_enhancer = bool(args.use_enhancer and args.mode == "purity_queries")
    model = HarmonicSlotTF(
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        in_channels=args.channels,
        dim=args.dim,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
        n_voices=3,
        mode=args.mode,
        legacy_slots=False,
        use_inpaint=use_inpaint,
        use_enhancer=use_enhancer,
        grad_checkpoint=args.grad_checkpoint,
        arch_version=args.arch,
    ).to(device)

    fft_sizes = (256, 512, 1024) if args.sr <= 24000 else (512, 1024, 2048)
    criterion = HybridSeparationLoss(
        model.bands,
        args.n_fft,
        args.hop_length,
        n_voices=3,
        activity_weight=args.activity_weight,
        ortho_weight=args.ortho_weight,
        bleed_weight=args.bleed_weight,
        cross_bleed_weight=args.cross_bleed_weight,
        tf_bleed_weight=args.tf_bleed_weight,
        fast_pit=args.fast_pit or (args.arch == "v3"),
        mix_cons_weight=args.mix_cons_weight,
        fft_sizes=fft_sizes,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_step = 0
    lr_base_step = 0
    best_sdr = -1e9
    best_score = -1e9
    trained_streaming = bool(args.streaming_prob > 0.0)
    ckpt: dict = {}
    if args.resume and args.resume.exists():
        if args.resume.suffix.lower() == ".safetensors":
            from safetensors import safe_open
            from safetensors.torch import load_file

            sd = load_file(str(args.resume), device=device)
            model.load_state_dict(sd)
            with safe_open(str(args.resume), framework="pt") as sf_f:
                meta = sf_f.metadata() or {}
            ckpt = {
                "step": int(meta.get("step", 0)),
                "best_sdr": float(meta.get("best_sdr", -1e9)),
                "trained_streaming": str(meta.get("trained_streaming", "False")) == "True",
            }
        else:
            try:
                ckpt = torch.load(args.resume, map_location=device, weights_only=True)
            except Exception:
                ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
        trained_streaming = trained_streaming or bool(ckpt.get("trained_streaming", False))
        start_step = int(ckpt.get("step", 0))
        if args.reset_refiners:
            _reset_refiners(model)
            lr_base_step = start_step
            print(f"Resumed base weights from {args.resume} at step {start_step} with refiners/optimizer/best_sdr reset")
        elif args.finetune:
            lr_base_step = start_step
            print(f"Stage-2 fine-tune resumed weights from {args.resume} at step {start_step} (fresh optimizer, best_sdr reset)")
        else:
            if ckpt.get("optimizer"):
                optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scaler") and use_scaler:
                scaler.load_state_dict(ckpt["scaler"])
            best_sdr = float(ckpt.get("best_sdr", -1e9))
            best_score = float(ckpt.get("best_score", best_sdr))
            lr_base_step = int(ckpt.get("lr_base_step", 0))
            print(f"Resumed from {args.resume} at step {start_step} (best_sdr={best_sdr:.2f} dB, lr_base_step={lr_base_step})")

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0.0 else None
    if ema is not None:
        if (
            args.resume
            and args.resume.exists()
            and not args.reset_refiners
            and not args.finetune
            and ckpt.get("ema")
        ):
            ema.load_state_dict(ckpt["ema"], device)
            print(f"Resumed EMA state (ema_updates={ema.num_updates})")
        elif args.finetune and start_step > 0:
            ema.num_updates = 100

    model.train()
    step = start_step
    accum = max(1, args.grad_accum)
    min_seg_samples = int(round(args.sr * args.seg_seconds))
    max_seg_samples = int(round(args.sr * max(args.seg_seconds, args.seg_seconds_max)))
    stream_c_len = int(round(args.sr * args.stream_chunk_seconds))
    stream_ov_len = int(round(args.sr * args.stream_overlap_seconds))
    stream_need_len = stream_c_len + max(1, stream_c_len - stream_ov_len)
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    micro_idx = 0
    while step < args.steps:
        for mix, targets, _ in train_loader:
            if step >= args.steps:
                break
            mix = mix.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            do_stream = (
                args.streaming_prob > 0.0
                and random.random() < args.streaming_prob
                and mix.shape[-1] >= stream_need_len
            )
            if do_stream:
                V = model.n_voices
                if random.random() < 0.40 or args.stream_chunk_seconds <= 4.0:
                    c_len = stream_c_len
                else:
                    c_len = int(round(args.sr * 4.0))
                    c_len = max(args.hop_length * 8, (c_len // args.hop_length) * args.hop_length)
                ov_cur = min(stream_ov_len, c_len // 2)
                ov_paused = [False] * mix.shape[0]
                is_gap_step = (random.random() < 0.50) and (mix.shape[-1] >= 2 * c_len)
                if is_gap_step:
                    max_gap = min(int(round(args.sr * 3.0)), mix.shape[-1] - 2 * c_len)
                    gap = random.randint(0, max(0, max_gap))
                    start2 = c_len + gap
                else:
                    start2 = c_len - ov_cur
                    if random.random() < 0.20 and ov_cur > 0:
                        start_ov = c_len - ov_cur
                        ov_w = c_len - start_ov
                        fade_n = min(int(0.015 * args.sr), max(1, ov_w // 4))
                        win_env = torch.zeros(ov_w, device=targets.device, dtype=targets.dtype)
                        if fade_n > 0:
                            t_f = torch.linspace(0.0, math.pi, fade_n, device=targets.device, dtype=targets.dtype)
                            win_env[:fade_n] = 0.5 * (1.0 + torch.cos(t_f))
                            win_env[-fade_n:] = 0.5 * (1.0 - torch.cos(t_f))
                        for b in range(mix.shape[0]):
                            act_v = [v for v in range(V) if targets[b, v, :, :start_ov].pow(2).mean().sqrt() > 1e-3]
                            if act_v:
                                dv = random.choice(act_v)
                                targets[b, dv, :, start_ov:c_len] = targets[b, dv, :, start_ov:c_len] * win_env
                                ov_paused[b] = True

                tgt1 = targets[..., :c_len].clone()
                tgt2 = targets[..., start2 : start2 + c_len].clone()

                def _keep_window(w_cn: torch.Tensor, keep_n: int, at_end: bool) -> torch.Tensor:
                    L = w_cn.shape[-1]
                    if keep_n >= L:
                        return w_cn
                    env = torch.zeros(L, device=w_cn.device, dtype=w_cn.dtype)
                    s0 = (L - keep_n) if at_end else 0
                    s1 = s0 + keep_n
                    env[s0:s1] = 1.0
                    fn = min(int(0.015 * args.sr), max(1, keep_n // 6))
                    if fn > 0:
                        tf = torch.linspace(0.0, math.pi, fn, device=w_cn.device, dtype=w_cn.dtype)
                        if s0 > 0:
                            env[s0 : s0 + fn] = 0.5 * (1.0 - torch.cos(tf))
                        if s1 < L:
                            env[s1 - fn : s1] = 0.5 * (1.0 + torch.cos(tf))
                    return w_cn * env

                for b in range(mix.shape[0]):
                    act_vs = [
                        v for v in range(V)
                        if tgt1[b, v].pow(2).mean().sqrt() > 1e-3 and tgt2[b, v].pow(2).mean().sqrt() > 1e-3
                    ]
                    if not act_vs:
                        continue
                    u = random.random()
                    if u < 0.05:
                        # 0-speaker gap chunk in Chunk 2 (music + ambience only while prev_slots is loaded)
                        tgt2[b, :V] = 0.0
                    elif u < 0.15:
                        # 1-speaker solo return across gap
                        kv = random.choice(act_vs)
                        for v in range(V):
                            if v != kv:
                                tgt1[b, v] = 0.0
                                tgt2[b, v] = 0.0
                        if random.random() < 0.50:
                            k_n = int(round(args.sr * random.uniform(1.8, min(3.0, c_len / args.sr))))
                            if random.random() < 0.50:
                                w_c = _keep_window(tgt1[b, kv], k_n, at_end=(random.random() < 0.5))
                                if w_c.pow(2).mean().sqrt() > 1e-3:
                                    tgt1[b, kv] = w_c
                            else:
                                w_c = _keep_window(tgt2[b, kv], k_n, at_end=(random.random() < 0.5))
                                if w_c.pow(2).mean().sqrt() > 1e-3:
                                    tgt2[b, kv] = w_c
                    elif u < 0.30 and len(act_vs) >= 2:
                        # Turn-taking: speaker dropout in Chunk 2 OR new speaker entering in Chunk 2
                        tv = random.choice(act_vs)
                        if random.random() < 0.50:
                            tgt2[b, tv] = 0.0
                        else:
                            tgt1[b, tv] = 0.0
                    elif not ov_paused[b] and random.random() < 0.25:
                        # Dense 2-3 speaker with a short 1.5s-2.5s returning utterance in Chunk 2 (+ optional 2-3.5s in Chunk 1)
                        rv = random.choice(act_vs)
                        if random.random() < 0.50:
                            k_n1 = int(round(args.sr * random.uniform(2.0, min(3.5, c_len / args.sr))))
                            w_c1 = _keep_window(tgt1[b, rv], k_n1, at_end=(random.random() < 0.5))
                            if w_c1.pow(2).mean().sqrt() > 1e-3:
                                tgt1[b, rv] = w_c1
                        k_n2 = int(round(args.sr * random.uniform(1.5, min(2.5, c_len / args.sr))))
                        w_c2 = _keep_window(tgt2[b, rv], k_n2, at_end=(random.random() < 0.5))
                        if w_c2.pow(2).mean().sqrt() > 1e-3:
                            tgt2[b, rv] = w_c2
                        gain = 10.0 ** (random.uniform(-3.5, 3.5) / 20.0)
                        tgt2[b, rv] = tgt2[b, rv] * gain

                mix1 = tgt1.sum(dim=1)
                mix2 = tgt2.sum(dim=1)
                for b in range(mix.shape[0]):
                    p1 = float(mix1[b].abs().max())
                    if p1 > 0.95:
                        sc1 = 0.95 / p1
                        tgt1[b] = tgt1[b] * sc1
                        mix1[b] = mix1[b] * sc1
                    p2 = float(mix2[b].abs().max())
                    if p2 > 0.95:
                        sc2 = 0.95 / p2
                        tgt2[b] = tgt2[b] * sc2
                        mix2[b] = mix2[b] * sc2
                c1_act = tgt1[:, :V].pow(2).mean(dim=(2, 3)).sqrt() > 1e-4
                c2_act = tgt2[:, :V].pow(2).mean(dim=(2, 3)).sqrt() > 1e-4
                with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                    out1 = model(mix1)
                    loss1, stats1 = criterion(out1, tgt1)
                    perms1 = criterion.last_perms
                    carry = _prepare_carry_slots(model, out1.get("slots"), perms1, tgt1)
                l1_val = float(loss1.detach())
                scaler.scale(0.5 * loss1 / float(accum)).backward()
                del out1, loss1
                locked_slots: List[Dict[int, int]] = []
                for b in range(mix.shape[0]):
                    locks: Dict[int, int] = {}
                    for idx_p, idx_t in enumerate(perms1[b]):
                        if bool(c1_act[b, idx_t]):
                            locks[idx_p] = idx_t
                    locked_slots.append(locks)
                with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                    out2 = model(mix2, prev_slots=carry)
                    loss2, stats2 = criterion(out2, tgt2, locked_slots=locked_slots)
                    slot_id_loss = loss2.new_zeros(())
                    if (
                        carry is not None
                        and out2.get("slots") is not None
                        and hasattr(model, "slot_module")
                        and hasattr(model.slot_module, "voice_queries")
                    ):
                        s2_v = out2["slots"][:, :V].float()
                        with torch.no_grad():
                            bq = model.slot_module.voice_queries.to(device=s2_v.device, dtype=torch.float32)
                            carry_f = carry.float()
                            delta1 = torch.nn.functional.normalize(carry_f - bq, dim=-1)
                            carry_norm = torch.nn.functional.normalize(carry_f, dim=-1)
                            if hasattr(model.slot_module, "prev_gate"):
                                bq_n = bq.norm(dim=-1, keepdim=True).clamp_min(1e-4)
                                ps_n = carry_f.norm(dim=-1, keepdim=True).clamp_min(1e-4)
                                ps_c = carry_f * (bq_n / ps_n)
                                pg_in = torch.cat(
                                    [ps_c, bq.expand_as(ps_c)], dim=-1
                                ).to(dtype=model.slot_module.prev_gate.weight.dtype)
                                g_c = torch.sigmoid(model.slot_module.prev_gate(pg_in).float())
                                v_slots2 = g_c * ps_c + (1.0 - g_c) * bq
                            else:
                                v_slots2 = bq
                        s2_norm = torch.nn.functional.normalize(s2_v, dim=-1)
                        delta2 = torch.nn.functional.normalize(s2_v - v_slots2, dim=-1)
                        id_terms: List[torch.Tensor] = []
                        for b in range(mix.shape[0]):
                            for idx_p, idx_t in enumerate(perms1[b]):
                                if not (bool(c1_act[b, idx_t]) and bool(c2_act[b, idx_t])):
                                    continue
                                align_full = 1.0 - (s2_norm[b, idx_p] * carry_norm[b, idx_p]).sum()
                                align_ac = 1.0 - (delta2[b, idx_p] * delta1[b, idx_p]).sum()
                                rep_terms = [
                                    (delta2[b, idx_p] * delta1[b, idx_q]).sum().square()
                                    for idx_q, idx_tq in enumerate(perms1[b])
                                    if idx_q != idx_p and bool(c1_act[b, idx_tq])
                                ]
                                rep_ac = sum(rep_terms) / len(rep_terms) if rep_terms else s2_v.new_zeros(())
                                id_terms.append(0.4 * align_full + 0.4 * align_ac + 0.2 * rep_ac)
                        if id_terms:
                            slot_id_loss = torch.stack(id_terms).mean()
                            loss2 = loss2 + 0.25 * slot_id_loss
                    stats2["slot_id"] = float(slot_id_loss.detach())
                    stats1["slot_id"] = stats2["slot_id"]
                l2_val = float(loss2.detach())
                scaler.scale(0.5 * loss2 / float(accum)).backward()
                loss_val = 0.5 * (l1_val + l2_val)
                del out2, loss2
                stats = {k: 0.5 * (stats1.get(k, 0.0) + stats2.get(k, 0.0)) for k in stats2}
            else:
                if max_seg_samples > min_seg_samples and mix.shape[-1] >= min_seg_samples:
                    crop_len = random.randint(min_seg_samples, min(max_seg_samples, mix.shape[-1]))
                else:
                    crop_len = min(min_seg_samples, mix.shape[-1])
                if crop_len < mix.shape[-1]:
                    offset = random.randint(0, mix.shape[-1] - crop_len)
                    mix = mix[..., offset : offset + crop_len]
                    targets = targets[..., offset : offset + crop_len]

                with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                    out = model(mix)
                    loss, stats = criterion(out, targets)
                loss_val = float(loss.detach())
                scaler.scale(loss / float(accum)).backward()
                del out, loss

            micro_idx += 1
            if micro_idx % accum != 0:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            total_sched = max(1, args.steps - lr_base_step)
            rel_step = step - lr_base_step
            warmup_steps = min(500, max(1, total_sched // 20))
            if rel_step < warmup_steps:
                cur_lr = args.lr * float(rel_step + 1) / float(warmup_steps)
            else:
                prog = min(1.0, float(rel_step - warmup_steps) / float(max(1, total_sched - warmup_steps)))
                cur_lr = args.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog)))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

            step += 1
            if step % 50 == 0 or step == args.steps:
                dt = time.time() - t0
                t0 = time.time()
                s_str = ", ".join(f"{k}={v:.2f}" for k, v in stats.items())
                print(f"step {step}/{args.steps} | loss={loss_val:.3f} ({s_str}) | {dt:.1f}s")

            if step % args.eval_every == 0 or step == args.steps:
                live_sd_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if ema is not None:
                    model.load_state_dict(ema.export_state_dict(model), strict=True)
                val = evaluate(
                    model, val_2_loader, val_3_loader, device, amp_dtype, amp_enabled=amp_enabled,
                    loader_ov2=val_ov2_loader, loader_ov3=val_ov3_loader, loader_stream=val_stream_loader,
                    stream_chunk_seconds=args.stream_chunk_seconds, stream_overlap_seconds=args.stream_overlap_seconds,
                    use_prev_slots=trained_streaming,
                )
                mean_voice_sdr = 0.25 * (
                    val["sisdr_2spk"] + val["sisdr_3spk"] + val["sisdr_ov2"] + val["sisdr_ov3"]
                )
                select_score = (
                    mean_voice_sdr - 2.0 * val["swap_rate"] - 2.0 * val["gap_swap_rate"]
                    if args.streaming_prob > 0.0
                    else mean_voice_sdr
                )
                print(
                    f"[VAL step {step}] 2spk={val['sisdr_2spk']:.2f} dB (i={val['sisdri_2spk']:+.2f}) | "
                    f"3spk={val['sisdr_3spk']:.2f} dB (i={val['sisdri_3spk']:+.2f}) | "
                    f"ov2={val['sisdr_ov2']:.2f} | ov3={val['sisdr_ov3']:.2f} | sir={val['sir_bleed']:.1f} dB | "
                    f"music={val['sisdr_music']:.2f} | amb={val['sisdr_amb']:.2f} | "
                    f"cnt_acc={val['count_acc']*100:.1f}% | swap={val['swap_rate']*100:.1f}% | "
                    f"gap_swap={val['gap_swap_rate']*100:.1f}% | "
                    f"scale={val['voice_scale']*100:.1f}% | unused={val['unused_slot_dbfs']:.1f} dBFS"
                )
                is_better = select_score > best_score
                new_best_sdr = mean_voice_sdr if is_better else best_sdr
                new_best_score = max(best_score, select_score)
                base_meta = {
                    "step": step,
                    "arch_version": args.arch,
                    "mode": args.mode,
                    "sr": args.sr,
                    "n_fft": args.n_fft,
                    "hop_length": args.hop_length,
                    "in_channels": args.channels,
                    "dim": args.dim,
                    "n_enc_layers": args.n_enc_layers,
                    "n_dec_layers": args.n_dec_layers,
                    "n_voices": 3,
                    "legacy_slots": False,
                    "use_inpaint": use_inpaint,
                    "use_enhancer": use_enhancer,
                    "trained_streaming": trained_streaming,
                    "best_sdr": new_best_sdr,
                }
                eval_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} if ema is not None else live_sd_cpu
                if ema is not None:
                    model.load_state_dict({k: v.to(device) for k, v in live_sd_cpu.items()}, strict=True)
                comfy_ckpt = {**base_meta, "model": eval_sd}
                full_ckpt = {
                    **base_meta,
                    "best_score": new_best_score,
                    "model": live_sd_cpu,
                    "ema": ema.state_dict() if ema is not None else None,
                    "lr_base_step": lr_base_step,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_scaler else None,
                }
                save_checkpoint(args.save_dir / "last.pt", full_ckpt)
                save_checkpoint(args.save_dir / "latest_comfy.pt", comfy_ckpt)
                save_safetensors(args.save_dir / "latest_comfy.safetensors", eval_sd, base_meta, cast_bf16=args.save_bf16)
                if is_better:
                    best_sdr = new_best_sdr
                    best_score = new_best_score
                    save_checkpoint(args.save_dir / "best.pt", comfy_ckpt)
                    save_safetensors(args.save_dir / "best.safetensors", eval_sd, base_meta, cast_bf16=args.save_bf16)


if __name__ == "__main__":
    main()
