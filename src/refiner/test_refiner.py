"""Double-audit CUDA verification & RTX 4060 benchmark for CrossStemDebleedRefiner v2."""
from __future__ import annotations

import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from refiner_model import CrossStemDebleedRefiner


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    model = CrossStemDebleedRefiner(
        sr=48000, n_fft=2048, hop_length=256, in_channels=2, n_voices=3,
        voice_base_c=48, bg_base_c=32, voice_bottleneck=3, bg_bottleneck=2,
        wave_hidden=32, grad_checkpoint=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    v_params = sum(p.numel() for p in model.voice_net.parameters())
    b_params = sum(p.numel() for p in model.bg_net.parameters())
    w_params = sum(p.numel() for p in model.voice_wave_net.parameters())
    print(
        f"Total params: {n_params:,} (voice_net={v_params:,}, bg_net={b_params:,}, wave_net={w_params:,}) "
        f"-> {n_params * 2 / 1e6:.2f} MB bf16"
    )

    # 1. Identity-init pass-through & silent-slot check (48 kHz, 2.0s)
    model.eval()
    N = 48000 * 2
    torch.manual_seed(0)
    raw = torch.randn(1, 5, 2, N, device=device) * 0.05
    raw[:, 2] = 0.0
    mix = raw.sum(dim=1)
    with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
        out = model(raw, mix)["stems"]
    assert out.shape == raw.shape
    silent_max = float(out[:, 2].abs().max())
    assert silent_max == 0.0, f"Silent slot leaked: {silent_max}"
    cos_v0 = float(torch.nn.functional.cosine_similarity(out[0, 0].flatten(), raw[0, 0].flatten(), dim=0))
    cos_m = float(torch.nn.functional.cosine_similarity(out[0, 3].flatten(), raw[0, 3].flatten(), dim=0))
    print(f"[1] Identity-init PASS: cos(v0)={cos_v0:.5f}, cos(music)={cos_m:.5f}, silent_v2_max={silent_max:.1e}")
    assert cos_v0 > 0.995 and cos_m > 0.995

    # 2. Strict Voice-Slot Permutation Equivariance check (swapping V0<->V2 must produce identical swapped output!)
    perm_idx = [2, 0, 1, 3, 4]
    raw_p = raw[:, perm_idx].contiguous()
    with torch.no_grad():
        out_fp32 = model(raw, mix)["stems"]
        out_p_fp32 = model(raw_p, mix)["stems"]
    inv_idx = [1, 2, 0, 3, 4]
    perm_diff = float((out_fp32 - out_p_fp32[:, inv_idx]).abs().max())
    print(f"[2] Permutation-equivariance PASS: max_diff={perm_diff:.2e}")
    assert perm_diff < 1e-5, f"Permutation equivariance violated: {perm_diff}"

    # 3. Exact Eval (per-stem loop) vs Train (batched) numerical equivalence in FP32
    model.train()
    with torch.no_grad():
        out_tr_fp32 = model(raw, mix)["stems"]
    model.eval()
    et_diff = float((out_fp32 - out_tr_fp32).abs().max())
    print(f"[3] Eval vs Train parity PASS: max_diff={et_diff:.2e}")
    assert et_diff < 1e-4, f"Eval/Train mismatch: {et_diff}"

    # 4. 100% Non-zero Gradient Flow check in bf16 WITH grad_checkpoint=True (B=1 4.0s exact L4 micro-batch!)
    model.train()
    N_tr = 48000 * 4
    raw_t = torch.randn(1, 5, 2, N_tr, device=device) * 0.05
    raw_t[:, 2] = 0.0
    mix_t = raw_t.sum(dim=1)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
        out_t = model(raw_t, mix_t)["stems"]
        tgt_t = raw_t + 0.05 * torch.randn_like(raw_t)
        loss = (out_t - tgt_t).abs().mean()
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().max() > 0)
    n_total = sum(1 for p in model.parameters())
    print(f"[4] bf16 + grad_ckpt Grad check PASS: {n_with_grad}/{n_total} param tensors received non-zero gradients")
    assert n_with_grad == n_total, f"Missing grads: {n_with_grad}/{n_total}"
    if device == "cuda":
        print(f"    Train B=1 4.0s (grad_ckpt=True) peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB (reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB)")
    model.zero_grad(set_to_none=True)
    del raw_t, mix_t, out_t, tgt_t, loss
    if device == "cuda":
        torch.cuda.empty_cache()

    # 5. Pause & solo augmentation + silent-pause zero re-injection + loss zero-NaN sanity check
    from train_refiner import _augment_pause_and_solo, _bg_voice_bleed_loss
    _PARENT = os.path.dirname(_HERE)
    if _PARENT not in sys.path:
        sys.path.insert(0, _PARENT)
    from loss import HybridSeparationLoss
    from model import build_variable_bands

    t_aug = torch.randn(8, 5, 2, 48000 * 4) * 0.1
    m_aug, t_out_aug = _augment_pause_and_solo(t_aug, sr=48000, n_voices=3)
    assert m_aug.shape == (8, 2, 48000 * 4) and t_out_aug.shape == t_aug.shape
    assert float(m_aug.abs().max()) <= 0.9501, f"Peak exceeded 0.95: {m_aug.abs().max()}"
    assert float((m_aug - t_out_aug.sum(dim=1)).abs().max()) < 1e-5, "Mix/target sum mismatch after aug!"

    # Verify local-RMS wave residual does NOT leak into a scrubbed silent pause
    from refiner_model import _apply_bounded_wave_residual
    base_p = torch.randn(1, 2, 48000 * 2, device=device) * 0.1
    base_p[..., 24000:72000] = 0.0  # 1.0s scrubbed silent pause
    raw_dw = torch.ones_like(base_p) * 10.0  # adversarial max tanh output during pause
    pol = _apply_bounded_wave_residual(base_p, raw_dw, max_ratio=0.15)
    pause_mid_max = float(pol[..., 24500:71500].abs().max())
    assert pause_mid_max < 1e-6, f"Wave residual leaked into silent pause: {pause_mid_max}"

    # Verify HybridSeparationLoss + _bg_voice_bleed_loss have zero NaN on 1-speaker solo + 2-speaker + 3-speaker batch
    bands = build_variable_bands(2048 // 2 + 1, sr=48000, n_fft=2048)
    crit = HybridSeparationLoss(
        bands, 2048, 256, n_voices=3, purity_weight=0.0, activity_weight=0.0, ortho_weight=0.0,
        bleed_weight=3.0, cross_bleed_weight=2.5, tf_bleed_weight=1.0, fast_pit=True, mix_cons_weight=0.1,
    ).to(device)
    t_edge = torch.randn(3, 5, 2, 48000, device=device) * 0.08
    t_edge[0, 1:3] = 0.0  # 1-speaker solo clip
    t_edge[1, 2] = 0.0    # 2-speaker clip with carved pause on speaker 0
    t_edge[1, 0, :, 12000:36000] = 0.0
    p_edge = (t_edge + 0.01 * torch.randn_like(t_edge)).requires_grad_(True)
    l_main, _ = crit({"stems": p_edge}, t_edge, fixed_perms=[(0, 1, 2)] * 3)
    l_bg = _bg_voice_bleed_loss(p_edge, t_edge, 2048, 256, torch.hann_window(2048, device=device))
    l_tot = l_main + 1.5 * l_bg
    assert torch.isfinite(l_tot), f"Non-finite loss on edge batch: {l_tot}"
    l_tot.backward()
    assert p_edge.grad is not None and torch.isfinite(p_edge.grad).all(), "Non-finite grad on edge batch!"
    print(f"[5] Aug + pause-zero-leak ({pause_mid_max:.1e}) + solo/pause loss finite PASS")

    # 6. RTX 4060 Inference VRAM & latency benchmark on 4.0s and 6.0s 48 kHz stereo
    model.eval()
    for sec in (4.0, 6.0):
        L = int(48000 * sec)
        x = torch.randn(1, 5, 2, L, device=device) * 0.05
        m = x.sum(dim=1)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            y = model(x, m)["stems"]
        if device == "cuda":
            torch.cuda.synchronize()
            pk = torch.cuda.max_memory_allocated() / 1e9
            rs = torch.cuda.max_memory_reserved() / 1e9
        else:
            pk = rs = 0.0
        dt = time.time() - t0
        assert y.shape == x.shape
        print(f"[6] Eval [{sec:.1f}s 48kHz stereo]: {dt:.2f}s | PEAK alloc={pk:.2f} GB | reserved={rs:.2f} GB")

    print("ALL 6 DOUBLE-AUDIT REFINER TESTS PASSED!")


if __name__ == "__main__":
    main()
