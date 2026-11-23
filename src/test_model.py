"""Verification for 3-speaker HarmonicSlotTF, 3 ablation modes, and HybridSeparationLoss."""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model import HarmonicSlotTF
from loss import HybridSeparationLoss


def test_if_dev_carrier_fix(device: str) -> None:
    m = HarmonicSlotTF(mode="plain").to(device)
    sr, n_fft, k_bin = 48000, 2048, 20
    t = torch.arange(sr * 2, device=device, dtype=torch.float32) / sr
    tone = torch.sin(2.0 * math.pi * (k_bin * sr / n_fft) * t).view(1, 1, -1).expand(1, 2, -1)
    X_btf = m._stft(tone).permute(0, 3, 2, 1)
    _, if_dev, _, _ = m._compute_physics(X_btf)
    interior_err = if_dev[0, 10:-10, k_bin].abs().max().item()
    print(f"IF carrier-fix check (tone @ bin {k_bin}): max |if_dev| = {interior_err:.2e}")
    assert interior_err < 1e-3


def test_activity_weighted_consistency(device: str) -> None:
    """Verify gated-off voice slot receives ~0 residual from mixture consistency."""
    m = HarmonicSlotTF(mode="purity_queries", n_voices=3).to(device)
    with torch.no_grad():
        m.activity_head.bias.data.fill_(0.0)
        # Force voice slot 2 off by hooking activity_head output
        orig_forward = m.activity_head.forward

        def forced_act(h):
            out = orig_forward(h)
            out[:, 2:3] = -20.0  # sigmoid(-20) ~ 2e-9
            return out

        m.activity_head.forward = forced_act  # type: ignore[method-assign]
        x = torch.randn(1, 2, 48000, device=device) * 0.2
        out = m(x)
        slot2_max = out["stems"][0, 2].abs().max().item()
        mix_err = (out["stems"].sum(dim=1) - x).abs().max().item()
        print(f"Gated-off slot 2 max amp: {slot2_max:.2e} | Mix err: {mix_err:.2e}")
        assert slot2_max < 1e-6, f"Gated-off slot leaked residual: {slot2_max}"


def bench_mode(mode: str, device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    m = HarmonicSlotTF(
        mode=mode,
        n_voices=3,
        use_inpaint=(mode == "purity_queries"),
        use_enhancer=(mode == "purity_queries"),
    ).to(device)  # type: ignore[arg-type]
    crit = HybridSeparationLoss(m.bands, m.n_fft, m.hop_length, m.n_voices).to(device)
    n_params = sum(p.numel() for p in m.parameters())

    # Batch=2, 5 stems (3 voices, music, ambience), 1.5s @ 48kHz (fits cleanly under 8GB VRAM), voice slot 2 silent
    targets = torch.randn(2, 5, 2, 72000, device=device) * 0.1
    targets[:, 2] = 0.0
    mix = targets.sum(dim=1)

    last_stats = {}

    def run_step() -> float:
        nonlocal last_stats
        m.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            out = m(mix)
            loss, last_stats = crit(out, targets)
        loss.backward()
        val = float(loss.detach())
        del out, loss
        return val

    run_step()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    loss_val = run_step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) * 1000.0
    vram = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
    bad_grads = [
        n for n, p in m.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    stats_str = ", ".join(f"{k}={v:.2f}" for k, v in last_stats.items())
    print(
        f"[{mode:<14}] Params: {n_params/1e6:.2f}M | Total: {loss_val:.2f} ({stats_str}) | "
        f"Grads ok: {len(bad_grads) == 0} | {dt:.1f} ms | {vram:.1f} MB"
    )
    if bad_grads:
        print("  Bad grads:", bad_grads[:5])
    del m, crit, targets, mix
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def test_purity_gate_gradient(device: str) -> None:
    """Verify voice purity heads get gradient directly from separation loss (no softmax cancel)."""
    m = HarmonicSlotTF(mode="purity_queries", n_voices=3).to(device)
    crit = HybridSeparationLoss(m.bands, m.n_fft, m.hop_length, m.n_voices, purity_weight=0.0).to(device)
    targets = torch.randn(1, 5, 2, 48000, device=device) * 0.1
    out = m(targets.sum(dim=1))
    loss, _ = crit(out, targets)
    loss.backward()
    cls_g = m.slot_module.class_head[1].weight.grad.abs().max().item()
    solo_g = m.slot_module.solo_spk_head[1].weight.grad.abs().max().item()
    print(f"Purity grad from separation loss only: class_head={cls_g:.2e}, solo_spk_head={solo_g:.2e}")
    assert cls_g > 0 and solo_g > 0


def test_zero_silence(device: str) -> None:
    """Verify exact zero input stays finite with zero energy-consistency NaNs."""
    m = HarmonicSlotTF(mode="purity_queries", n_voices=3).to(device)
    m.eval()
    with torch.no_grad():
        x = torch.zeros(1, 2, 48000, device=device)
        out = m(x)
        assert torch.isfinite(out["stems"]).all(), "NaN on zero input!"
        err = out["stems"].abs().max().item()
        print(f"Zero-silence input max output: {err:.2e}")
        assert err < 1e-6


def test_pairwise_upit_equivalence(device: str) -> None:
    """Verify 3x3 pairwise uPIT matches brute-force permutation loop."""
    import itertools

    m = HarmonicSlotTF(sr=16000, n_fft=512, hop_length=128, in_channels=1, mode="plain").to(device)
    crit = HybridSeparationLoss(m.bands, m.n_fft, m.hop_length, m.n_voices, fft_sizes=(256, 512, 1024)).to(device)
    targets = torch.randn(2, 5, 1, 16000, device=device) * 0.1
    out = m(targets.sum(dim=1))
    _, stats = crit(out, targets)

    preds = out["stems"]
    brute = torch.stack([
        sum(crit._stem_loss(preds[:, i], targets[:, j]) for i, j in enumerate(perm))
        for perm in itertools.permutations(range(3))
    ], dim=1).min(dim=1).values.mean().item()
    diff = abs(stats["voice"] - brute)
    print(f"3x3 uPIT vs brute-force max diff: {diff:.2e}")
    assert diff < 1e-5


def test_v2_slot_mechanics_and_grads(device: str) -> None:
    """Verify v2 orthogonal voice queries, comp_tk conditioning, stem_proj, and band_pos stay distinct with grads."""
    import torch.nn.functional as F

    m = HarmonicSlotTF(sr=16000, n_fft=512, hop_length=128, in_channels=1, mode="purity_queries").to(device)
    crit = HybridSeparationLoss(
        m.bands, m.n_fft, m.hop_length, m.n_voices, fft_sizes=(256, 512, 1024)
    ).to(device)
    targets = torch.randn(1, 5, 1, 16000, device=device) * 0.1
    out = m(targets.sum(dim=1))
    slots = out["slots"]
    assert slots is not None and slots.shape == (1, 5, 128)
    vq = F.normalize(m.slot_module.voice_queries[0], dim=-1)
    init_cos = (vq @ vq.t() - torch.eye(3, device=device)).abs().max().item()
    vs = F.normalize(slots[0, :3], dim=-1)
    ext_cos = (vs @ vs.t() - torch.eye(3, device=device)).max().item()
    print(f"Initial voice query |cos|={init_cos:.2e} | Extracted voice slot max cos={ext_cos:.3f}")
    assert init_cos < 1e-5
    assert ext_cos < 0.5, f"Extracted voice slots collapsed: max_cos={ext_cos}"

    loss, stats = crit(out, targets)
    assert "ortho" in stats and stats["ortho"] < 0.2, f"Unexpected ortho loss at step 0: {stats}"
    loss.backward()
    assert m.band_pos.grad is not None and m.band_pos.grad.abs().max().item() > 0
    assert m.slot_module.voice_queries.grad is not None and m.slot_module.voice_queries.grad.abs().max().item() > 0
    assert m.slot_module.stem_proj.weight.grad is not None and m.slot_module.stem_proj.weight.grad.abs().max().item() > 0
    print(
        f"v2 grads ok: stem_proj_g={m.slot_module.stem_proj.weight.grad.abs().max().item():.2e}, "
        f"ortho={stats['ortho']:.3f}"
    )


def test_legacy_phase1_checkpoints() -> None:
    """Verify existing Phase 1 checkpoints in ComfyUI/models/audiosep load cleanly with strict=True."""
    import pathlib

    comfy_root = pathlib.Path(r"d:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI")
    ckpt_root = comfy_root / "models" / "audiosep" / "audiosep_16k"
    if not ckpt_root.exists():
        print("Skipping legacy checkpoint check (audiosep_16k dir absent)")
        return
    sys.path.insert(0, str(comfy_root))
    node_dir = str(comfy_root / "custom_nodes" / "ComfyUI-HarmonicSlotTF")
    sys.path.insert(0, node_dir)
    import importlib.util

    spec = importlib.util.spec_from_file_location("hstf_nodes", str(pathlib.Path(node_dir) / "nodes.py"))
    assert spec and spec.loader
    # Load via package import so relative import `.harmonic_slot_tf` resolves
    sys.path.insert(0, str(comfy_root / "custom_nodes"))
    hstf_nodes = importlib.import_module("ComfyUI-HarmonicSlotTF.nodes")

    for sub in ("plain", "queries", "purity_queries"):
        bak_path = ckpt_root / sub / "best.v1_bak.pt"
        ckpt_path = bak_path if bak_path.exists() else (ckpt_root / sub / "best.pt")
        if not ckpt_path.exists():
            continue
        sd, meta = hstf_nodes._load_checkpoint_data(str(ckpt_path))
        cfg = hstf_nodes._infer_model_config(sd, meta)
        assert cfg["legacy_slots"] is True, f"Expected legacy_slots=True for {ckpt_path}"
        m = hstf_nodes.HarmonicSlotTF(**cfg)
        m.load_state_dict(sd, strict=True)
        print(f"Loaded Phase 1 ckpt {sub}/{ckpt_path.name} cleanly (legacy_slots={cfg['legacy_slots']})")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    test_if_dev_carrier_fix(device)
    test_activity_weighted_consistency(device)
    test_zero_silence(device)
    test_purity_gate_gradient(device)
    test_pairwise_upit_equivalence(device)
    test_v2_slot_mechanics_and_grads(device)
    test_legacy_phase1_checkpoints()
    for mode in ("plain", "queries", "purity_queries"):
        bench_mode(mode, device)


if __name__ == "__main__":
    main()
