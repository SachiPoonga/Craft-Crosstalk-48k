"""Hybrid loss for 2-3 Voices + Music + Ambience separation.

- uPIT across V voice slots (silent stems handled cleanly without SC blowup).
- Fixed-head loss for Music & Ambience.
- Decoupled 2-head purity supervision on energy (mag^2) with clip-scaled eps:
    1) class_logits: [speech, music, ambience] energy fractions.
    2) solo_spk_logits: max_v E_v / (sum_v E_v + eps_clip), masked where speech is active.
- Voice activity BCE loss.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResSTFTLoss(nn.Module):
    def __init__(self, fft_sizes: Tuple[int, ...] = (512, 1024, 2048), tf_bleed_weight: float = 0.25):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.tf_bleed_weight = tf_bleed_weight
        for n_fft in fft_sizes:
            self.register_buffer(f"win_{n_fft}", torch.hann_window(n_fft), persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: [B, C, N] -> returns [B] per-sample loss."""
        B, C, N = pred.shape
        p = pred.reshape(B * C, N).float()
        t = target.reshape(B * C, N).float()
        active = (t.pow(2).mean(dim=1).sqrt() > 1e-4).float()
        total = 0.0
        for n_fft in self.fft_sizes:
            win = getattr(self, f"win_{n_fft}").to(p.device)
            hop = n_fft // 4
            Pc = torch.stft(p, n_fft, hop, n_fft, win, center=True, return_complex=True)
            Tc = torch.stft(t, n_fft, hop, n_fft, win, center=True, return_complex=True)
            P = Pc.abs()
            T = Tc.abs()
            t_norm = T.flatten(1).norm(dim=1)
            sc = ((T - P).flatten(1).norm(dim=1) / t_norm.clamp_min(1e-4)) * active
            P_pow = (P + 1e-8).pow(0.3)
            T_pow = (T + 1e-8).pow(0.3)
            mag = (torch.log1p(T) - torch.log1p(P)).abs().mean(dim=(1, 2)) + 1.5 * (T_pow - P_pow).abs().mean(dim=(1, 2))
            t_mean = T.mean(dim=(1, 2), keepdim=True).clamp_min(1e-4)
            t_bin_thresh = torch.clamp(0.02 * t_mean, min=1e-4)
            tf_silent = (T < t_bin_thresh).float()
            tf_bleed = ((P / t_mean) * tf_silent).mean(dim=(1, 2)) * active
            Pc_comp = torch.view_as_real(Pc / (P + 1e-5) * P_pow)
            Tc_comp = torch.view_as_real(Tc / (T + 1e-5) * T_pow)
            c_loss = (Pc_comp - Tc_comp).abs().mean(dim=(1, 2, 3))
            total = total + (sc + mag + 0.5 * c_loss + self.tf_bleed_weight * tf_bleed)
        return (total / len(self.fft_sizes)).reshape(B, C).mean(dim=1)


class HybridSeparationLoss(nn.Module):
    """uPIT over voices + fixed Music/Ambience + decoupled 2-head purity & activity."""

    def __init__(
        self,
        bands: List[Tuple[int, int]],
        n_fft: int = 2048,
        hop_length: int = 256,
        n_voices: int = 3,
        purity_weight: float = 0.1,
        activity_weight: float = 0.1,
        ortho_weight: float = 0.2,
        bleed_weight: float = 1.0,
        cross_bleed_weight: float = 0.0,
        tf_bleed_weight: float = 0.25,
        fast_pit: bool = False,
        mix_cons_weight: float = 0.0,
        fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
    ):
        super().__init__()
        self.bands = bands
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_voices = n_voices
        self.purity_weight = purity_weight
        self.activity_weight = activity_weight
        self.ortho_weight = ortho_weight
        self.bleed_weight = bleed_weight
        self.cross_bleed_weight = cross_bleed_weight
        self.fast_pit = fast_pit
        self.mix_cons_weight = mix_cons_weight
        self.last_perms: List[Tuple[int, ...]] = []
        self.mrstft = MultiResSTFTLoss(fft_sizes=fft_sizes, tf_bleed_weight=tf_bleed_weight)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def _sisdr_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Scale-invariant SDR loss [B] combining whole-clip and 0.25s-0.5s segmental windows."""
        B, C, N = pred.shape
        p = pred.reshape(B, -1).float()
        t = target.reshape(B, -1).float()
        active = (t.pow(2).mean(dim=1).sqrt() > 1e-4).float()
        p_z = p - p.mean(dim=1, keepdim=True)
        t_z = t - t.mean(dim=1, keepdim=True)
        t_energy = t_z.pow(2).sum(dim=1, keepdim=True) + 1e-8
        proj = ((p_z * t_z).sum(dim=1, keepdim=True) / t_energy) * t_z
        noise = p_z - proj
        eps_e = 1e-6 * t_energy.squeeze(1)
        sisdr = 10.0 * torch.log10((proj.pow(2).sum(dim=1) + 1e-30) / (noise.pow(2).sum(dim=1) + eps_e))
        whole_loss = active * (-sisdr.clamp(min=-30.0, max=30.0)) / 10.0

        win_len = max(2048, self.hop_length * 32)
        if N < win_len:
            return whole_loss

        step = win_len // 2
        p_bc = pred.reshape(B * C, N).float()
        t_bc = target.reshape(B * C, N).float()
        p_w = p_bc.unfold(-1, win_len, step)
        t_w = t_bc.unfold(-1, win_len, step)
        w_t_rms = t_w.pow(2).mean(dim=-1).sqrt()
        clip_t_rms = t_bc.pow(2).mean(dim=1, keepdim=True).sqrt()
        w_active = (w_t_rms > torch.clamp(0.05 * clip_t_rms, min=1e-4)).float()

        p_w = p_w - p_w.mean(dim=-1, keepdim=True)
        t_w = t_w - t_w.mean(dim=-1, keepdim=True)
        w_t_energy = t_w.pow(2).sum(dim=-1, keepdim=True) + 1e-8
        w_proj = ((p_w * t_w).sum(dim=-1, keepdim=True) / w_t_energy) * t_w
        w_noise = p_w - w_proj
        w_eps_e = 1e-6 * w_t_energy.squeeze(-1)
        w_sisdr = 10.0 * torch.log10((w_proj.pow(2).sum(dim=-1) + 1e-30) / (w_noise.pow(2).sum(dim=-1) + w_eps_e))
        seg_loss = (
            (w_active * (-w_sisdr.clamp(min=-30.0, max=30.0) / 10.0)).sum(dim=-1)
            / w_active.sum(dim=-1).clamp_min(1.0)
        ).reshape(B, C).mean(dim=1)
        return 0.4 * whole_loss + 0.6 * seg_loss

    def _silent_frame_bleed_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Penalize RMS + scale-normalized leakage during frames where target speaker is silent [B]."""
        B, C, N = pred.shape
        frame_len = max(256, self.hop_length * 2)
        if N < frame_len:
            return torch.zeros(B, device=pred.device, dtype=torch.float32)
        step = frame_len // 2
        p_frames = pred.reshape(B * C, N).float().unfold(-1, frame_len, step)
        t_frames = target.reshape(B * C, N).float().unfold(-1, frame_len, step)
        p_rms = p_frames.pow(2).mean(dim=-1).add(1e-12).sqrt()
        t_rms = t_frames.pow(2).mean(dim=-1).sqrt()
        clip_t_rms = target.reshape(B * C, N).float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        silent_mask = (t_rms < torch.clamp(0.02 * clip_t_rms, min=1e-4)).float()
        bleed_frame = (p_rms + 0.02 * (p_rms / (clip_t_rms + 1e-3))) * silent_mask
        return bleed_frame.mean(dim=-1).reshape(B, C).mean(dim=1)

    def _stem_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = (pred - target).abs().mean(dim=(1, 2))
        stft = self.mrstft(pred, target)
        bleed = self._silent_frame_bleed_loss(pred, target)
        p_rms = pred.float().pow(2).mean(dim=(1, 2)).add(1e-12).sqrt()
        t_rms = target.float().pow(2).mean(dim=(1, 2)).add(1e-12).sqrt()
        active = (t_rms > 1e-4).float()
        scale_loss = active * (torch.log(p_rms + 1e-4) - torch.log(t_rms + 1e-4)).abs()
        return l1 + 0.5 * stft + 1.0 * self._sisdr_loss(pred, target) + self.bleed_weight * bleed + 0.5 * scale_loss

    @torch.no_grad()
    def _fast_pair_cost(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Fast whole-clip L1 + negative SI-SDR cost [B] for uPIT matching without MRSTFT overhead."""
        B = pred.shape[0]
        p = pred.reshape(B, -1).float()
        t = target.reshape(B, -1).float()
        l1 = (p - t).abs().mean(dim=1)
        active = (t.pow(2).mean(dim=1).sqrt() > 1e-4).float()
        pz = p - p.mean(dim=1, keepdim=True)
        tz = t - t.mean(dim=1, keepdim=True)
        te = tz.pow(2).sum(dim=1, keepdim=True) + 1e-8
        proj = ((pz * tz).sum(dim=1, keepdim=True) / te) * tz
        noise = pz - proj
        sisdr = 10.0 * torch.log10((proj.pow(2).sum(dim=1) + 1e-30) / (noise.pow(2).sum(dim=1) + 1e-6 * te.squeeze(1)))
        bleed = self._silent_frame_bleed_loss(pred, target)
        p_rms = p.pow(2).mean(dim=1).add(1e-12).sqrt()
        t_rms = t.pow(2).mean(dim=1).add(1e-12).sqrt()
        scale_loss = active * (torch.log(p_rms + 1e-4) - torch.log(t_rms + 1e-4)).abs()
        return l1 + active * (-sisdr.clamp(min=-30.0, max=30.0)) / 10.0 + self.bleed_weight * bleed + 0.5 * scale_loss

    def _cross_speaker_bleed_loss(self, matched_preds: torch.Tensor, targets_v: torch.Tensor) -> torch.Tensor:
        """Penalize interferer-dominant STFT leakage and pause-overlap energy leakage."""
        B, V, C, N = matched_preds.shape
        if V <= 1:
            return matched_preds.new_zeros(())
        p = matched_preds.float()
        t = targets_v.float()
        active = (t.pow(2).mean(dim=(2, 3)).sqrt() > 1e-4).float()  # [B, V]
        pair_active = active.unsqueeze(2) * active.unsqueeze(1) * (
            1.0 - torch.eye(V, device=p.device).unsqueeze(0)
        )
        n_pairs = pair_active.sum().clamp_min(1.0)

        # 1) Pure interferer-dominant STFT magnitude leakage (normalized over dom bins only)
        win = self.window.to(p.device)
        P = torch.stft(p.reshape(B * V * C, N), self.n_fft, self.hop_length, self.n_fft, win, center=True, return_complex=True).abs()
        T = torch.stft(t.reshape(B * V * C, N), self.n_fft, self.hop_length, self.n_fft, win, center=True, return_complex=True).abs()
        _, F_b, T_f = P.shape
        P = P.reshape(B, V, C, F_b, T_f)
        T = T.reshape(B, V, C, F_b, T_f)
        t_mean = T.mean(dim=(2, 3, 4), keepdim=True).clamp_min(1e-4)  # [B, V, 1, 1, 1]
        stft_bleed = p.new_zeros(())
        for i in range(V):
            for j in range(V):
                if i == j:
                    continue
                dom = ((T[:, j] > 0.10 * t_mean[:, j]) & (T[:, i] < 0.15 * T[:, j])).float()
                dom_sum = dom.sum(dim=(1, 2, 3))
                leak_ratio = (P[:, i] / (T[:, j] + 0.05 * t_mean[:, j])).clamp(max=2.0)
                pair_stft = ((leak_ratio * dom).sum(dim=(1, 2, 3)) / dom_sum.clamp_min(1.0)) * (dom_sum > 0).float()
                stft_bleed = stft_bleed + (pair_stft * pair_active[:, i, j]).sum()
        stft_bleed = stft_bleed / n_pairs

        # 2) Phase-invariant ~128 ms pause-overlap energy leakage
        win_len = max(2048, self.hop_length * 24)
        win_bleed = p.new_zeros(())
        if N >= win_len:
            step = win_len // 2
            pw = p.unfold(-1, win_len, step)  # [B, V, C, W, L]
            tw = t.unfold(-1, win_len, step)
            p_w_rms = pw.pow(2).mean(dim=(2, 4)).add(1e-12).sqrt()  # [B, V, W]
            t_w_rms = tw.pow(2).mean(dim=(2, 4)).add(1e-12).sqrt()  # [B, V, W]
            clip_rms = t.pow(2).mean(dim=(2, 3)).add(1e-12).sqrt().unsqueeze(-1)  # [B, V, 1]
            for i in range(V):
                for j in range(V):
                    if i == j:
                        continue
                    w_mask = ((t_w_rms[:, i] < 0.05 * clip_rms[:, i]) & (t_w_rms[:, j] > 0.15 * clip_rms[:, j])).float()
                    w_sum = w_mask.sum(dim=-1)
                    w_leak = (p_w_rms[:, i] / (t_w_rms[:, j] + 1e-4)).clamp(max=2.0)
                    pair_win = ((w_leak * w_mask).sum(dim=-1) / w_sum.clamp_min(1.0)) * (w_sum > 0).float()
                    win_bleed = win_bleed + (pair_win * pair_active[:, i, j]).sum()
            win_bleed = win_bleed / n_pairs

        return 0.65 * stft_bleed + 0.35 * win_bleed

    @torch.no_grad()
    def _decoupled_purity_targets(
        self, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ground-truth class fractions & single-speaker dominance on energy (mag^2).

        Returns:
          cls_tgt:       [B, T, K, 3]  ([speech, music, ambience])
          solo_tgt:      [B, T, K, 1]  (max_v E_v / sum_v E_v)
          speech_active: [B, T, K, 1]  mask where speech energy is non-negligible
        """
        B, S, C, N = targets.shape
        V = self.n_voices
        X = torch.stft(
            targets.reshape(B * S * C, N).float(),
            self.n_fft,
            self.hop_length,
            self.n_fft,
            self.window.to(targets.device),
            center=True,
            return_complex=True,
        )
        # Energy (|X|^2) averaged across channels -> [B, T, F, S]
        energy = X.abs().pow(2).reshape(B, S, C, X.shape[-2], X.shape[-1]).mean(dim=2)
        energy = energy.permute(0, 3, 2, 1)
        band_e = torch.stack(
            [energy[:, :, i:j, :].sum(dim=2) for (i, j) in self.bands], dim=2
        )  # [B, T, K, S]

        speech_e = band_e[..., :V].sum(dim=-1, keepdim=True)   # [B, T, K, 1]
        music_e = band_e[..., V : V + 1]
        amb_e = band_e[..., V + 1 : V + 2]
        cls_e = torch.cat([speech_e, music_e, amb_e], dim=-1)   # [B, T, K, 3]

        # Scale eps to each clip's mean energy so quiet vs loud clips behave identically
        clip_scale = cls_e.sum(dim=-1, keepdim=True).mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        eps_cls = 1e-4 * clip_scale
        cls_tgt = cls_e / (cls_e.sum(dim=-1, keepdim=True) + eps_cls)

        max_v = band_e[..., :V].max(dim=-1, keepdim=True).values
        solo_tgt = max_v / (speech_e + eps_cls)
        speech_active = (speech_e > 1e-3 * clip_scale).float()
        return cls_tgt, solo_tgt, speech_active

    def forward(
        self,
        out: Dict[str, torch.Tensor | None],
        targets: torch.Tensor,
        fixed_perms: List[Tuple[int, ...]] | None = None,
        locked_slots: List[Dict[int, int]] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        preds = out["stems"]
        assert preds is not None
        B, S, C, N = preds.shape
        V = self.n_voices

        # 1. Fixed loss for Music & Ambience
        fixed_loss = 0.0
        for s in range(V, S):
            fixed_loss = fixed_loss + self._stem_loss(preds[:, s], targets[:, s]).mean()

        # 2. Voice slot matching (fixed_perms, partial locked_slots for Phase-2 streaming, fast_pit, or full uPIT)
        perms = list(itertools.permutations(range(V)))
        if fixed_perms is not None:
            chosen_perms = [tuple(int(x) for x in p) for p in fixed_perms]
            matched_preds = torch.zeros_like(preds[:, :V])
            for b in range(B):
                for idx_p, idx_t in enumerate(chosen_perms[b]):
                    matched_preds[b, idx_t] = preds[b, idx_p]
            voice_loss = sum(self._stem_loss(matched_preds[:, j], targets[:, j]) for j in range(V)).mean()
        elif self.fast_pit or locked_slots is not None:
            pair_costs = [
                [self._fast_pair_cost(preds[:, i], targets[:, j]) for j in range(V)]
                for i in range(V)
            ]
            perm_costs = torch.stack(
                [
                    sum(pair_costs[idx_p][idx_t] for idx_p, idx_t in enumerate(perm))
                    for perm in perms
                ],
                dim=1,
            )  # [B, V!]
            if locked_slots is not None:
                for b in range(B):
                    locks = locked_slots[b]
                    if not locks:
                        continue
                    allowed = [
                        p_idx
                        for p_idx, perm in enumerate(perms)
                        if all(perm[idx_p] == idx_t for idx_p, idx_t in locks.items())
                    ]
                    if allowed:
                        mask = torch.ones(len(perms), device=perm_costs.device, dtype=torch.bool)
                        mask[allowed] = False
                        perm_costs[b, mask] = float("inf")
            _, best_idx = perm_costs.min(dim=1)
            chosen_perms = [perms[int(best_idx[b])] for b in range(B)]
            matched_preds = torch.zeros_like(preds[:, :V])
            for b in range(B):
                for idx_p, idx_t in enumerate(chosen_perms[b]):
                    matched_preds[b, idx_t] = preds[b, idx_p]
            voice_loss = sum(self._stem_loss(matched_preds[:, j], targets[:, j]) for j in range(V)).mean()
        else:
            pair_losses = [
                [self._stem_loss(preds[:, i], targets[:, j]) for j in range(V)]
                for i in range(V)
            ]
            perm_losses = [
                sum(pair_losses[idx_p][idx_t] for idx_p, idx_t in enumerate(perm))
                for perm in perms
            ]
            best_voice_loss, best_idx = torch.stack(perm_losses, dim=1).min(dim=1)
            chosen_perms = [perms[int(best_idx[b])] for b in range(B)]
            voice_loss = best_voice_loss.mean()
            matched_preds = None

        self.last_perms = chosen_perms

        total = voice_loss + fixed_loss
        stats: Dict[str, float] = {
            "voice": float(voice_loss.detach()),
            "fixed": float(fixed_loss.detach()),
        }

        if self.cross_bleed_weight > 0.0 and V > 1:
            if matched_preds is None:
                matched_preds = torch.zeros_like(preds[:, :V])
                for b in range(B):
                    for idx_p, idx_t in enumerate(chosen_perms[b]):
                        matched_preds[b, idx_t] = preds[b, idx_p]
            cross_bleed = self._cross_speaker_bleed_loss(matched_preds, targets[:, :V])
            total = total + self.cross_bleed_weight * cross_bleed
            stats["xbleed"] = float(cross_bleed.detach())

        if self.mix_cons_weight > 0.0:
            mix_pred = preds.float().sum(dim=1)
            mix_tgt = targets.float().sum(dim=1)
            mix_l1 = (mix_pred - mix_tgt).abs().mean()
            win = self.window.to(mix_pred.device)
            Mp = torch.stft(mix_pred.reshape(B * C, N), self.n_fft, self.hop_length, self.n_fft, win, center=True, return_complex=True).abs()
            Mt = torch.stft(mix_tgt.reshape(B * C, N), self.n_fft, self.hop_length, self.n_fft, win, center=True, return_complex=True).abs()
            mix_cons = mix_l1 + 0.25 * (torch.log1p(Mp) - torch.log1p(Mt)).abs().mean()
            total = total + self.mix_cons_weight * mix_cons
            stats["mcons"] = float(mix_cons.detach())

        # 3. Permutation-invariant voice activity loss
        if out.get("activity_logits") is not None:
            rms = targets[:, :V].pow(2).mean(dim=(2, 3)).sqrt() > 1e-4  # [B, V]
            active = torch.zeros(B, V, device=targets.device)
            for b in range(B):
                for idx_p, idx_t in enumerate(chosen_perms[b]):
                    active[b, idx_p] = float(rms[b, idx_t])
            act_loss = F.binary_cross_entropy_with_logits(out["activity_logits"][:, :V], active)
            total = total + self.activity_weight * act_loss
            stats["activity"] = float(act_loss.detach())

        # 4. Decoupled 2-head purity supervision (no permutation ambiguity!)
        if out.get("class_logits") is not None and out.get("solo_spk_logits") is not None:
            cls_tgt, solo_tgt, speech_active = self._decoupled_purity_targets(targets)
            log_cls_pred = F.log_softmax(out["class_logits"], dim=-1)
            cls_loss = -(cls_tgt * log_cls_pred).sum(dim=-1).mean()

            solo_bce = F.binary_cross_entropy_with_logits(
                out["solo_spk_logits"], solo_tgt, reduction="none"
            )
            solo_loss = (solo_bce * speech_active).sum() / speech_active.sum().clamp_min(1.0)
            total = total + self.purity_weight * (cls_loss + solo_loss)
            stats["cls_pur"] = float(cls_loss.detach())
            stats["solo_pur"] = float(solo_loss.detach())

        # 5. Voice slot cosine orthogonality / repulsion to prevent speaker slot collapse
        if out.get("slots") is not None and V > 1:
            v_slots = F.normalize(out["slots"][:, :V].float(), dim=-1)
            sim = torch.bmm(v_slots, v_slots.transpose(1, 2))
            eye = torch.eye(V, device=sim.device, dtype=sim.dtype).unsqueeze(0)
            ortho_loss = ((sim * (1.0 - eye)).square().sum(dim=(1, 2)) / max(1, V * (V - 1))).mean()
            total = total + self.ortho_weight * ortho_loss
            stats["ortho"] = float(ortho_loss.detach())

        return total, stats
