"""Train CrossStemDebleedRefiner chained after a frozen HarmonicSlot-TF v3 teacher.

Key guarantees:
  - Frozen teacher runs in eval() + torch.no_grad() + bf16 (uses VRAM-efficient inference).
  - Exact same leak-proof disjoint train/val speaker split (seed=42) as train.py.
  - Strict slot alignment: matches teacher voice outputs to ground-truth speakers via
    fast PIT, reorders ground-truth targets to match teacher slot order, and trains
    the refiner with fixed_perms=[(0, 1, 2)] so Phase-2 slot locking is 100% preserved.
  - Aggressive anti-bleed losses: silent-frame bleed, TF silent-bin bleed, cross-speaker
    interferer-dominant bleed, AND vocal-in-music/ambience bleed suppression.
  - Logs side-by-side BEFORE (frozen teacher) vs AFTER (+ refiner) metrics on every eval.
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
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PARENT)

from loss import HybridSeparationLoss
from refiner_model import CrossStemDebleedRefiner
from train import (
    DynamicMixDataset,
    ModelEMA,
    SyntheticMixDataset,
    _compute_batch_sir,
    _discover_files,
    _discover_speakers,
    save_checkpoint,
    save_safetensors,
    sisdr,
)

# Prefer the memory-efficient inference HarmonicSlotTF if ComfyUI node copy exists, else AudioSep/model.py
_COMFY_MODEL_DIR = os.path.join(
    os.path.dirname(_PARENT),
    "ComfyUI_windows_portable",
    "ComfyUI",
    "custom_nodes",
    "ComfyUI-HarmonicSlotTF",
)
if os.path.isfile(os.path.join(_COMFY_MODEL_DIR, "harmonic_slot_tf.py")):
    sys.path.insert(0, _COMFY_MODEL_DIR)
    from harmonic_slot_tf import HarmonicSlotTF  # type: ignore
else:
    from model import HarmonicSlotTF  # type: ignore


def _load_frozen_teacher(ckpt_path: Path, device: str) -> Tuple[HarmonicSlotTF, dict]:
    """Load frozen HarmonicSlot-TF v3 teacher from .safetensors or .pt."""
    if ckpt_path.suffix.lower() == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        sd = load_file(str(ckpt_path), device=device)
        with safe_open(str(ckpt_path), framework="pt") as f:
            raw_meta = f.metadata() or {}
        meta = {
            "sr": int(raw_meta.get("sr", 48000)),
            "n_fft": int(raw_meta.get("n_fft", 2048)),
            "hop_length": int(raw_meta.get("hop_length", 256)),
            "in_channels": int(raw_meta.get("in_channels", 2)),
            "dim": int(raw_meta.get("dim", 384)),
            "n_enc_layers": int(raw_meta.get("n_enc_layers", 8)),
            "n_dec_layers": int(raw_meta.get("n_dec_layers", 4)),
            "n_voices": int(raw_meta.get("n_voices", 3)),
            "mode": str(raw_meta.get("mode", "purity_queries")),
            "arch_version": str(raw_meta.get("arch_version", "v3")),
            "use_inpaint": str(raw_meta.get("use_inpaint", "True")) == "True",
            "use_enhancer": str(raw_meta.get("use_enhancer", "True")) == "True",
            "legacy_slots": str(raw_meta.get("legacy_slots", "False")) == "True",
            "step": int(raw_meta.get("step", 0)),
        }
    else:
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt["model"]
        meta = {
            "sr": int(ckpt.get("sr", 48000)),
            "n_fft": int(ckpt.get("n_fft", 2048)),
            "hop_length": int(ckpt.get("hop_length", 256)),
            "in_channels": int(ckpt.get("in_channels", 2)),
            "dim": int(ckpt.get("dim", 384)),
            "n_enc_layers": int(ckpt.get("n_enc_layers", 8)),
            "n_dec_layers": int(ckpt.get("n_dec_layers", 4)),
            "n_voices": int(ckpt.get("n_voices", 3)),
            "mode": str(ckpt.get("mode", "purity_queries")),
            "arch_version": str(ckpt.get("arch_version", "v3")),
            "use_inpaint": bool(ckpt.get("use_inpaint", True)),
            "use_enhancer": bool(ckpt.get("use_enhancer", True)),
            "legacy_slots": bool(ckpt.get("legacy_slots", False)),
            "step": int(ckpt.get("step", 0)),
        }

    teacher = HarmonicSlotTF(
        sr=meta["sr"],
        n_fft=meta["n_fft"],
        hop_length=meta["hop_length"],
        in_channels=meta["in_channels"],
        dim=meta["dim"],
        n_enc_layers=meta["n_enc_layers"],
        n_dec_layers=meta["n_dec_layers"],
        n_voices=meta["n_voices"],
        mode=meta["mode"],
        legacy_slots=meta["legacy_slots"],
        use_inpaint=meta["use_inpaint"],
        use_enhancer=meta["use_enhancer"],
        grad_checkpoint=False,
        arch_version=meta["arch_version"],
    ).to(device)
    teacher.load_state_dict(sd, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, meta


@torch.no_grad()
def _align_targets_to_teacher(
    criterion: HybridSeparationLoss,
    raw_stems: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align ground-truth voice targets to teacher's slot order without permuting teacher outputs."""
    B, S, C, N = raw_stems.shape
    V = criterion.n_voices
    perms = list(itertools.permutations(range(V)))
    pair_costs = [
        [criterion._fast_pair_cost(raw_stems[:, i], targets[:, j]) for j in range(V)]
        for i in range(V)
    ]
    perm_costs = torch.stack(
        [sum(pair_costs[idx_p][idx_t] for idx_p, idx_t in enumerate(perm)) for perm in perms],
        dim=1,
    )
    _, best_idx = perm_costs.min(dim=1)
    aligned = targets.clone()
    active_mask = torch.zeros(B, V, device=raw_stems.device, dtype=torch.float32)
    for b in range(B):
        perm = perms[int(best_idx[b])]
        for idx_p, idx_t in enumerate(perm):
            aligned[b, idx_p] = targets[b, idx_t]
            is_act = (
                raw_stems[b, idx_p].float().pow(2).mean().sqrt() > 1e-4
                or targets[b, idx_t].float().pow(2).mean().sqrt() > 1e-4
            )
            active_mask[b, idx_p] = 1.0 if is_act else 0.0
    return aligned, active_mask


def _bg_voice_bleed_loss(
    refined_stems: torch.Tensor,
    aligned_targets: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor,
    n_voices: int = 3,
) -> torch.Tensor:
    """Penalize speech leakage inside refined Music & Ambience stems on speech-dominant STFT bins."""
    B, S, C, N = refined_stems.shape
    win = window.to(refined_stems.device)
    speech_tgt = aligned_targets[:, :n_voices].float().sum(dim=1)  # [B, C, N]
    bg_pred = refined_stems[:, n_voices:].float()                   # [B, 2, C, N]
    bg_tgt = aligned_targets[:, n_voices:].float()                  # [B, 2, C, N]

    T_sp = torch.stft(speech_tgt.reshape(B * C, N), n_fft, hop_length, n_fft, win, center=True, return_complex=True).abs()
    P_bg = torch.stft(bg_pred.reshape(B * 2 * C, N), n_fft, hop_length, n_fft, win, center=True, return_complex=True).abs()
    T_bg = torch.stft(bg_tgt.reshape(B * 2 * C, N), n_fft, hop_length, n_fft, win, center=True, return_complex=True).abs()

    _, F_b, T_f = T_sp.shape
    T_sp = T_sp.reshape(B, 1, C, F_b, T_f)
    P_bg = P_bg.reshape(B, 2, C, F_b, T_f)
    T_bg = T_bg.reshape(B, 2, C, F_b, T_f)

    sp_mean = T_sp.mean(dim=(2, 3, 4), keepdim=True).clamp_min(1e-4)
    dom = ((T_sp > 0.12 * sp_mean) & (T_bg < 0.15 * T_sp)).float()
    dom_sum = dom.sum(dim=(2, 3, 4))
    leak = (P_bg / (T_sp + 0.05 * sp_mean)).clamp(max=2.0)
    loss_per = ((leak * dom).sum(dim=(2, 3, 4)) / dom_sum.clamp_min(1.0)) * (dom_sum > 0).float()
    return loss_per.mean()


@torch.no_grad()
def evaluate_before_after(
    teacher: HarmonicSlotTF,
    refiner: CrossStemDebleedRefiner,
    loaders: Dict[str, Tuple[int, DataLoader, bool]],
    device: str,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate frozen teacher (BEFORE) and teacher+refiner (AFTER) on exact same val mixes."""
    refiner.eval()
    V = refiner.n_voices
    sir_win = torch.hann_window(refiner.n_fft, device=device)

    def _empty_m() -> Dict[str, List[float]]:
        return {
            "sisdr_2spk": [],
            "sisdr_3spk": [],
            "sisdr_ov2": [],
            "sisdr_ov3": [],
            "sir_bleed": [],
            "sisdr_music": [],
            "sisdr_amb": [],
            "voice_scale": [],
            "unused_slot_dbfs": [],
        }

    m_bef = _empty_m()
    m_aft = _empty_m()
    slot_flips = 0
    slot_total = 0

    for tag, (n_act, loader, compute_sir) in loaders.items():
        perms = list(itertools.permutations(range(V), n_act))
        for mix, targets, _ in loader:
            mix = mix.to(device)
            targets = targets.to(device)
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                t_out = teacher(mix)
                raw = t_out["stems"].float()
                r_out = refiner(raw, mix)
                ref = r_out["stems"].float()

            for b in range(mix.shape[0]):
                # Find teacher's slot permutation and evaluate BOTH raw & ref under the EXACT same slot mapping
                best_sdr_raw = -1e9
                best_perm = perms[0]
                for perm in perms:
                    ms = sum(sisdr(raw[b, perm[j]], targets[b, j]).mean().item() for j in range(n_act)) / n_act
                    if ms > best_sdr_raw:
                        best_sdr_raw = ms
                        best_perm = perm
                # Also verify refiner did not flip speaker slots
                best_perm_ref = max(
                    perms,
                    key=lambda p: sum(sisdr(ref[b, p[j]], targets[b, j]).mean().item() for j in range(n_act)),
                )
                slot_total += 1
                if best_perm_ref != best_perm:
                    slot_flips += 1

                for preds, store in ((raw, m_bef), (ref, m_aft)):
                    sdr_val = sum(sisdr(preds[b, best_perm[j]], targets[b, j]).mean().item() for j in range(n_act)) / n_act
                    store[f"sisdr_{tag}"].append(sdr_val)
                    if tag in ("2spk", "3spk"):
                        for j in range(n_act):
                            p_rms = float(preds[b, best_perm[j]].pow(2).mean().sqrt())
                            t_rms = float(targets[b, j].pow(2).mean().sqrt().clamp_min(1e-8))
                            store["voice_scale"].append(p_rms / t_rms)
                        if n_act == 2:
                            unused = next(idx for idx in range(V) if idx not in best_perm)
                            rms_u = float(preds[b, unused].pow(2).mean().sqrt().clamp_min(1e-10))
                            store["unused_slot_dbfs"].append(20.0 * math.log10(rms_u))
                        if targets[b, 3].abs().max() > 1e-4:
                            store["sisdr_music"].append(sisdr(preds[b, 3], targets[b, 3]).mean().item())
                        if targets[b, 4].abs().max() > 1e-4:
                            store["sisdr_amb"].append(sisdr(preds[b, 4], targets[b, 4]).mean().item())
                    if compute_sir:
                        store["sir_bleed"].extend(
                            _compute_batch_sir(
                                preds[b, :V], targets[b, :V], best_perm, n_act, refiner.n_fft, refiner.hop_length, sir_win
                            )
                        )

    refiner.train()
    bef = {k: (sum(v) / len(v) if v else 0.0) for k, v in m_bef.items()}
    aft = {k: (sum(v) / len(v) if v else 0.0) for k, v in m_aft.items()}
    aft["slot_flip_rate"] = float(slot_flips) / max(1, slot_total)
    return bef, aft


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CrossStemDebleedRefiner on frozen Craft Crosstalk-48k")
    p.add_argument("--teacher-ckpt", type=Path, default=None, help="Frozen HarmonicSlot-TF v3 .safetensors or .pt")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16", "auto"], default="bf16")
    p.add_argument("--save-bf16", action="store_true", default=True)
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--channels", type=int, default=2)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=256)
    p.add_argument("--voice-base-c", type=int, default=48)
    p.add_argument("--bg-base-c", type=int, default=32)
    p.add_argument("--voice-bottleneck", type=int, default=3)
    p.add_argument("--bg-bottleneck", type=int, default=2)
    p.add_argument("--wave-hidden", type=int, default=32)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seg-seconds", type=float, default=4.0)
    p.add_argument("--val-seg-seconds", type=float, default=4.0)
    p.add_argument("--p-3spk", type=float, default=0.55)
    p.add_argument("--hard-lang-prob", type=float, default=0.45)
    p.add_argument("--full-overlap-prob", type=float, default=0.55)
    p.add_argument("--mono-mix-prob", type=float, default=0.50)
    p.add_argument("--bleed-weight", type=float, default=3.0)
    p.add_argument("--tf-bleed-weight", type=float, default=1.0)
    p.add_argument("--cross-bleed-weight", type=float, default=2.5)
    p.add_argument("--bg-voice-bleed-weight", type=float, default=1.5)
    p.add_argument("--mix-cons-weight", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints_refiner"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--speech-dir", type=Path, default=None)
    p.add_argument("--music-dir", type=Path, default=None)
    p.add_argument("--amb-dir", type=Path, default=None)
    p.add_argument("--rir-dir", type=Path, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--synthetic-smoke", action="store_true")
    return p.parse_args()


def _augment_pause_and_solo(
    targets: torch.Tensor,
    sr: int,
    n_voices: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inject 10% solo-speaker clips and 25% sharp 1.5s-2.5s silent pauses to crush pause bleed."""
    B, S, C, N = targets.shape
    tgt = targets.clone()
    fade_n = min(int(0.015 * sr), max(1, N // 32))
    for b in range(B):
        act_vs = [v for v in range(n_voices) if tgt[b, v].pow(2).mean().sqrt() > 1e-3]
        if not act_vs:
            continue
        u = random.random()
        if u < 0.10:
            # 10%: 1-speaker solo clip (keeps solo voice pristine + unused slots dead silent)
            keep_v = random.choice(act_vs)
            for v in range(n_voices):
                if v != keep_v:
                    tgt[b, v] = 0.0
        elif u < 0.35 and len(act_vs) >= 2 and N >= int(2.5 * sr):
            # 25%: Carve a clean 1.5s-2.5s silent pause out of 1 active speaker while others talk
            pv = random.choice(act_vs)
            max_pause = min(int(2.5 * sr), int(0.60 * N))
            min_pause = min(int(1.2 * sr), max_pause)
            p_len = random.randint(min_pause, max_pause)
            p_start = random.randint(0, N - p_len)
            p_end = p_start + p_len
            env = torch.ones(N, device=tgt.device, dtype=tgt.dtype)
            env[p_start:p_end] = 0.0
            fn = min(fade_n, max(1, p_len // 6))
            if fn > 0:
                tf = torch.linspace(0.0, math.pi, fn, device=tgt.device, dtype=tgt.dtype)
                if p_start >= fn:
                    env[p_start - fn : p_start] = 0.5 * (1.0 + torch.cos(tf))
                if p_end + fn <= N:
                    env[p_end : p_end + fn] = 0.5 * (1.0 - torch.cos(tf))
            cand = tgt[b, pv] * env
            if cand.pow(2).mean().sqrt() > 1e-3:
                tgt[b, pv] = cand
    mix = tgt.sum(dim=1)
    for b in range(B):
        pk = float(mix[b].abs().max())
        if pk > 0.95:
            sc = 0.95 / pk
            tgt[b] = tgt[b] * sc
            mix[b] = mix[b] * sc
    return mix, tgt


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if args.precision == "fp32" or device != "cuda":
        amp_enabled, amp_dtype, use_scaler = False, torch.float32, False
    elif args.precision == "bf16":
        amp_enabled, amp_dtype, use_scaler = True, torch.bfloat16, False
    elif args.precision == "fp16":
        amp_enabled, amp_dtype, use_scaler = True, torch.float16, True
    else:
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_enabled = True
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        use_scaler = not use_bf16

    if args.synthetic_smoke:
        teacher = HarmonicSlotTF(
            sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length, in_channels=args.channels,
            dim=32, n_enc_layers=1, n_dec_layers=1, n_voices=3, mode="purity_queries",
            use_inpaint=True, use_enhancer=True, arch_version="v3",
        ).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        t_meta = {"sr": args.sr, "n_fft": args.n_fft, "hop_length": args.hop_length, "in_channels": args.channels, "step": 0}
        train_ds = SyntheticMixDataset(args.sr, args.channels, args.seg_seconds, length=32)
        val_2_ds = SyntheticMixDataset(args.sr, args.channels, args.val_seg_seconds, length=4, fixed_n_active=2)
        val_3_ds = SyntheticMixDataset(args.sr, args.channels, args.val_seg_seconds, length=4, fixed_n_active=3)
        val_ov2_ds = SyntheticMixDataset(args.sr, args.channels, args.val_seg_seconds, length=4, fixed_n_active=2)
        val_ov3_ds = SyntheticMixDataset(args.sr, args.channels, args.val_seg_seconds, length=4, fixed_n_active=3)
    else:
        assert args.teacher_ckpt and args.teacher_ckpt.exists(), f"Missing --teacher-ckpt: {args.teacher_ckpt}"
        teacher, t_meta = _load_frozen_teacher(args.teacher_ckpt, device)
        args.sr = t_meta["sr"]
        args.n_fft = t_meta["n_fft"]
        args.hop_length = t_meta["hop_length"]
        args.channels = t_meta["in_channels"]

        assert args.speech_dir and args.music_dir and args.amb_dir, "Provide --speech-dir, --music-dir, --amb-dir"
        spk_groups = _discover_speakers(args.speech_dir)
        spk_ids = sorted(spk_groups.keys())
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
            f"Teacher loaded (step={t_meta['step']}, sr={args.sr}) | "
            f"Train spks={len(trn_spks)}, Val spks={len(val_spks)}, Music={len(music_all)}, Amb={len(amb_all)}"
        )
        train_ds = DynamicMixDataset(
            trn_spks, music_all[n_vm:], amb_all[n_va:], trn_rirs,
            args.sr, args.channels, args.seg_seconds, n_samples_epoch=100000, seed=42,
            p_3spk=args.p_3spk, hard_lang_prob=args.hard_lang_prob,
            full_overlap_prob=args.full_overlap_prob, mono_mix_prob=args.mono_mix_prob,
        )
        val_2_ds = DynamicMixDataset(val_spks, music_all[:n_vm], amb_all[:n_va], val_rirs, args.sr, args.channels, args.val_seg_seconds, n_samples_epoch=24, seed=100, fixed_n_active=2, mono_mix_prob=args.mono_mix_prob)
        val_3_ds = DynamicMixDataset(val_spks, music_all[:n_vm], amb_all[:n_va], val_rirs, args.sr, args.channels, args.val_seg_seconds, n_samples_epoch=24, seed=200, fixed_n_active=3, mono_mix_prob=args.mono_mix_prob)
        val_ov2_ds = DynamicMixDataset(val_spks, music_all[:n_vm], amb_all[:n_va], val_rirs, args.sr, args.channels, args.val_seg_seconds, n_samples_epoch=24, seed=300, fixed_n_active=2, force_full_overlap=True, mono_mix_prob=args.mono_mix_prob)
        val_ov3_ds = DynamicMixDataset(val_spks, music_all[:n_vm], amb_all[:n_va], val_rirs, args.sr, args.channels, args.val_seg_seconds, n_samples_epoch=24, seed=400, fixed_n_active=3, force_full_overlap=True, mono_mix_prob=args.mono_mix_prob)

    nw = 0 if args.synthetic_smoke else args.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw,
        pin_memory=(device == "cuda"), persistent_workers=(nw > 0),
    )
    val_loaders = {
        "2spk": (2, DataLoader(val_2_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw), False),
        "3spk": (3, DataLoader(val_3_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw), False),
        "ov2": (2, DataLoader(val_ov2_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw), True),
        "ov3": (3, DataLoader(val_ov3_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw), True),
    }

    refiner = CrossStemDebleedRefiner(
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        in_channels=args.channels,
        n_voices=3,
        voice_base_c=args.voice_base_c,
        bg_base_c=args.bg_base_c,
        voice_bottleneck=args.voice_bottleneck,
        bg_bottleneck=args.bg_bottleneck,
        wave_hidden=args.wave_hidden,
        grad_checkpoint=args.grad_checkpoint,
    ).to(device)
    n_params = sum(p.numel() for p in refiner.parameters())
    print(
        f"CrossStemDebleedRefiner | Params={n_params:,} ({n_params * 2 / 1e6:.2f} MB bf16) | "
        f"Device={device} | Precision={args.precision}"
    )

    fft_sizes = (256, 512, 1024) if args.sr <= 24000 else (512, 1024, 2048)
    criterion = HybridSeparationLoss(
        teacher.bands,
        args.n_fft,
        args.hop_length,
        n_voices=3,
        purity_weight=0.0,
        activity_weight=0.0,
        ortho_weight=0.0,
        bleed_weight=args.bleed_weight,
        cross_bleed_weight=args.cross_bleed_weight,
        tf_bleed_weight=args.tf_bleed_weight,
        fast_pit=True,
        mix_cons_weight=args.mix_cons_weight,
        fft_sizes=fft_sizes,
    ).to(device)

    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    ema = ModelEMA(refiner, decay=args.ema_decay) if args.ema_decay > 0.0 else None

    start_step = 0
    best_score = -1e9
    if args.resume and args.resume.exists():
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        refiner.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and use_scaler:
            scaler.load_state_dict(ckpt["scaler"])
        if ema is not None and ckpt.get("ema"):
            ema.load_state_dict(ckpt["ema"], device)
        start_step = int(ckpt.get("step", 0))
        best_score = float(ckpt.get("best_score", -1e9))
        print(f"Resumed refiner from {args.resume} at step {start_step} (best_score={best_score:.3f})")

    # Print baseline BEFORE vs AFTER at step 0
    if start_step == 0 and not args.synthetic_smoke:
        bef, aft = evaluate_before_after(teacher, refiner, val_loaders, device, amp_dtype, amp_enabled)
        print(
            f"[BASELINE step 0] BEFORE: 2spk={bef['sisdr_2spk']:.2f} | 3spk={bef['sisdr_3spk']:.2f} | "
            f"ov2={bef['sisdr_ov2']:.2f} | ov3={bef['sisdr_ov3']:.2f} | sir={bef['sir_bleed']:.1f} dB | "
            f"music={bef['sisdr_music']:.2f} | amb={bef['sisdr_amb']:.2f}"
        )
        print(
            f"[BASELINE step 0] AFTER0: 2spk={aft['sisdr_2spk']:.2f} | 3spk={aft['sisdr_3spk']:.2f} | "
            f"ov2={aft['sisdr_ov2']:.2f} | ov3={aft['sisdr_ov3']:.2f} | sir={aft['sir_bleed']:.1f} dB | "
            f"music={aft['sisdr_music']:.2f} | amb={aft['sisdr_amb']:.2f}"
        )

    refiner.train()
    step = start_step
    accum = max(1, args.grad_accum)
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    micro_idx = 0
    loss_acc = 0.0
    fixed_win = torch.hann_window(args.n_fft, device=device)

    while step < args.steps:
        for mix, targets, _ in train_loader:
            if step >= args.steps:
                break
            mix = mix.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if not args.synthetic_smoke:
                mix, targets = _augment_pause_and_solo(targets, args.sr, n_voices=3)

            # 1. Frozen teacher inference (no grad)
            with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                t_out = teacher(mix)
                raw_stems = t_out["stems"].float()
            aligned_targets, active_mask = _align_targets_to_teacher(criterion, raw_stems, targets)
            fixed_perms = [(0, 1, 2)] * mix.shape[0]

            # 2. Refiner forward + aggressive anti-bleed loss under strict fixed slot order
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                out = refiner(raw_stems, mix, active_voice_mask=active_mask)
                loss, stats = criterion(out, aligned_targets, fixed_perms=fixed_perms)
                if args.bg_voice_bleed_weight > 0.0:
                    bg_vb = _bg_voice_bleed_loss(
                        out["stems"], aligned_targets, args.n_fft, args.hop_length, fixed_win, n_voices=3
                    )
                    loss = loss + args.bg_voice_bleed_weight * bg_vb
                    stats["bg_vb"] = float(bg_vb.detach())

            loss_acc += float(loss.detach()) / float(accum)
            scaler.scale(loss / float(accum)).backward()
            del t_out, raw_stems, out, loss

            micro_idx += 1
            if micro_idx % accum != 0:
                continue
            loss_val = loss_acc
            loss_acc = 0.0

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 5.0)
            warmup_steps = min(400, max(1, args.steps // 20))
            if step < warmup_steps:
                cur_lr = args.lr * float(step + 1) / float(warmup_steps)
            else:
                prog = min(1.0, float(step - warmup_steps) / float(max(1, args.steps - warmup_steps)))
                cur_lr = args.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog)))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(refiner)

            step += 1
            if step % 50 == 0 or step == args.steps:
                dt = time.time() - t0
                t0 = time.time()
                s_str = ", ".join(f"{k}={v:.2f}" for k, v in stats.items())
                print(f"step {step}/{args.steps} | loss={loss_val:.3f} ({s_str}) | lr={cur_lr:.2e} | {dt:.1f}s")

            if step % args.eval_every == 0 or step == args.steps:
                live_sd = {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()}
                if ema is not None:
                    refiner.load_state_dict(ema.export_state_dict(refiner), strict=True)
                bef, aft = evaluate_before_after(teacher, refiner, val_loaders, device, amp_dtype, amp_enabled)

                mean_sdr_aft = 0.25 * (
                    aft["sisdr_2spk"] + aft["sisdr_3spk"] + aft["sisdr_ov2"] + aft["sisdr_ov3"]
                )
                mean_sdr_bef = 0.25 * (
                    bef["sisdr_2spk"] + bef["sisdr_3spk"] + bef["sisdr_ov2"] + bef["sisdr_ov3"]
                )
                # Score rewards separation SDR + SIR bleed reduction + bg purity while guarding volume & slot-lock
                select_score = (
                    mean_sdr_aft
                    + 0.20 * aft["sir_bleed"]
                    + 0.10 * aft["sisdr_music"]
                    + 0.10 * aft["sisdr_amb"]
                    - 2.0 * abs(aft["voice_scale"] - 1.0)
                    - 5.0 * aft.get("slot_flip_rate", 0.0)
                )
                print(
                    f"[VAL step {step}] AFTER: "
                    f"2spk={aft['sisdr_2spk']:.2f} ({aft['sisdr_2spk']-bef['sisdr_2spk']:+.2f}) | "
                    f"3spk={aft['sisdr_3spk']:.2f} ({aft['sisdr_3spk']-bef['sisdr_3spk']:+.2f}) | "
                    f"ov2={aft['sisdr_ov2']:.2f} ({aft['sisdr_ov2']-bef['sisdr_ov2']:+.2f}) | "
                    f"ov3={aft['sisdr_ov3']:.2f} ({aft['sisdr_ov3']-bef['sisdr_ov3']:+.2f}) | "
                    f"sir={aft['sir_bleed']:.1f} dB ({aft['sir_bleed']-bef['sir_bleed']:+.1f}) | "
                    f"music={aft['sisdr_music']:.2f} ({aft['sisdr_music']-bef['sisdr_music']:+.2f}) | "
                    f"amb={aft['sisdr_amb']:.2f} ({aft['sisdr_amb']-bef['sisdr_amb']:+.2f}) | "
                    f"scale={aft['voice_scale']*100:.1f}% | unused={aft['unused_slot_dbfs']:.1f} dBFS | "
                    f"flip={aft.get('slot_flip_rate', 0.0)*100:.1f}% | "
                    f"mean_v={mean_sdr_aft:.2f} ({mean_sdr_aft-mean_sdr_bef:+.2f})"
                )

                is_better = select_score > best_score
                new_best = max(best_score, select_score)
                base_meta = {
                    "model_type": "CrossStemDebleedRefiner",
                    "refiner_version": "v2_5ref_wave",
                    "step": step,
                    "teacher_step": t_meta["step"],
                    "sr": args.sr,
                    "n_fft": args.n_fft,
                    "hop_length": args.hop_length,
                    "in_channels": args.channels,
                    "n_voices": 3,
                    "voice_base_c": args.voice_base_c,
                    "bg_base_c": args.bg_base_c,
                    "voice_bottleneck": args.voice_bottleneck,
                    "bg_bottleneck": args.bg_bottleneck,
                    "wave_hidden": refiner.wave_hidden,
                    "best_score": new_best,
                    "sir_bleed": aft["sir_bleed"],
                    "mean_voice_sdr": mean_sdr_aft,
                }
                eval_sd = {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()} if ema is not None else live_sd
                if ema is not None:
                    refiner.load_state_dict({k: v.to(device) for k, v in live_sd.items()}, strict=True)

                full_ckpt = {
                    **base_meta,
                    "model": live_sd,
                    "ema": ema.state_dict() if ema is not None else None,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_scaler else None,
                }
                save_checkpoint(args.save_dir / "last_refiner.pt", full_ckpt)
                save_safetensors(args.save_dir / "latest_refiner.safetensors", eval_sd, base_meta, cast_bf16=args.save_bf16)
                if is_better:
                    best_score = new_best
                    save_safetensors(args.save_dir / "best_refiner.safetensors", eval_sd, base_meta, cast_bf16=args.save_bf16)


if __name__ == "__main__":
    main()
