"""ComfyUI custom nodes for HarmonicSlotTF multi-stem audio separation."""
from __future__ import annotations

import itertools
import os
from typing import Any, Dict, Tuple

import torch
import torchaudio

import comfy.model_management
import folder_paths

from .harmonic_slot_tf import HarmonicSlotTF
from .refiner_model import CrossStemDebleedRefiner

_MODEL_EXTS = {".pt", ".pth", ".ckpt", ".safetensors"}
_AUDIOSEP_DIR = os.path.join(folder_paths.models_dir, "audiosep")
os.makedirs(_AUDIOSEP_DIR, exist_ok=True)
folder_paths.add_model_folder_path("audiosep", _AUDIOSEP_DIR, is_default=True)
if "audiosep" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["audiosep"][1].update(_MODEL_EXTS)

_EMPTY_PLACEHOLDER = "(put .pt or .safetensors in ComfyUI/models/audiosep)"

_SR_HOP_DEFAULTS: Dict[int, Tuple[int, int]] = {
    512: (16000, 128),
    1024: (32000, 256),
    1536: (48000, 384),
    2048: (48000, 256),
}


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _parse_safetensors_meta(raw: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if v == "True":
            out[k] = True
        elif v == "False":
            out[k] = False
        else:
            try:
                out[k] = int(v)
                continue
            except ValueError:
                pass
            try:
                out[k] = float(v)
                continue
            except ValueError:
                pass
            out[k] = v
    return out


def _load_checkpoint_data(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(path, framework="pt") as sf_f:
            meta = _parse_safetensors_meta(sf_f.metadata() or {})
        return _strip_module_prefix(load_file(path, device="cpu")), meta

    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        raw = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        meta = {k: v for k, v in raw.items() if k not in ("model", "optimizer", "scaler")}
        return _strip_module_prefix(raw["model"]), meta
    if isinstance(raw, dict):
        return _strip_module_prefix(raw), {}
    raise ValueError(f"Unsupported checkpoint payload in {path}")


def _infer_model_config(
    state_dict: Dict[str, torch.Tensor],
    meta: Dict[str, Any],
    sr_override: int = 0,
    hop_override: int = 0,
) -> Dict[str, Any]:
    """Infer HarmonicSlotTF constructor arguments directly from checkpoint weights."""
    mode = meta.get("mode")
    if mode is None:
        if "stem_proj.weight" in state_dict:
            mode = "plain"
        elif "slot_module.class_head.0.weight" in state_dict:
            mode = "purity_queries"
        else:
            mode = "queries"

    arch_version = str(
        meta.get("arch_version")
        or (
            "v3"
            if (
                "slot_module.inter_attn2.in_proj_weight" in state_dict
                or any(k.startswith("band_enc_groups.") for k in state_dict)
            )
            else "v2"
        )
    )

    if "dim" in meta:
        dim = int(meta["dim"])
    elif arch_version == "v3" and "band_enc_groups.0.b" in state_dict:
        dim = int(state_dict["band_enc_groups.0.b"].shape[-1])
    else:
        dim = int(state_dict["voice_decoders.0.weight"].shape[1])

    enc_indices = [
        int(k.split(".")[1])
        for k in state_dict
        if k.startswith("encoder.") and k.split(".")[1].isdigit()
    ]
    n_enc_layers = int(meta.get("n_enc_layers") or (max(enc_indices) + 1))

    dec_indices = [
        int(k.split(".")[1])
        for k in state_dict
        if k.startswith("decoder.") and k.split(".")[1].isdigit()
    ]
    n_dec_layers = int(meta.get("n_dec_layers") or (max(dec_indices) + 1 if dec_indices else 0))

    if "n_voices" in meta:
        n_voices = int(meta["n_voices"])
    elif "stem_proj.weight" in state_dict:
        n_voices = int(state_dict["stem_proj.weight"].shape[0] // dim - 2)
    elif "slot_module.voice_queries" in state_dict:
        n_voices = int(state_dict["slot_module.voice_queries"].shape[1])
    else:
        n_voices = 3

    if arch_version == "v3":
        band0_dec_out = int(state_dict["band_dec_groups.0.w_v"].shape[2])
        band0_enc_in = int(state_dict["band_enc_groups.0.w"].shape[1])
    else:
        band0_dec_out = int(state_dict["voice_decoders.0.weight"].shape[0])
        band0_enc_in = int(state_dict["band_encoders.0.0.weight"].shape[0])

    in_channels = meta.get("in_channels")
    if in_channels is None:
        for candidate_c in (2, 1):
            denom = 2 * candidate_c
            if band0_dec_out % denom != 0:
                continue
            bw0 = band0_dec_out // denom
            if bw0 * (2 * candidate_c + 3) + (dim // 2) == band0_enc_in:
                in_channels = candidate_c
                break
        if in_channels is None:
            raise ValueError(
                f"Could not infer in_channels from band0 shapes ({band0_enc_in}, {band0_dec_out})"
            )
    in_channels = int(in_channels)

    if arch_version == "v3":
        total_voice_out = sum(
            v.shape[0] * v.shape[2]
            for k, v in state_dict.items()
            if k.startswith("band_dec_groups.") and k.endswith(".w_v")
        )
    else:
        total_voice_out = sum(
            v.shape[0]
            for k, v in state_dict.items()
            if k.startswith("voice_decoders.") and k.endswith(".weight")
        )
    n_freqs = total_voice_out // (2 * in_channels)
    n_fft = int(meta.get("n_fft") or ((n_freqs - 1) * 2))

    default_sr, default_hop = _SR_HOP_DEFAULTS.get(
        n_fft, (48000 if n_fft >= 1024 else 16000, n_fft // 4)
    )
    sr = sr_override if sr_override > 0 else int(meta.get("sr") or default_sr)
    hop_length = hop_override if hop_override > 0 else int(meta.get("hop_length") or default_hop)
    legacy_slots = (
        ("slot_module.voice_mu" in state_dict)
        or ("band_pos" not in state_dict)
        or ("encoder.0.norm_t_attn.weight" not in state_dict)
    )
    use_inpaint = any(k.startswith("inpaint_ref.") for k in state_dict)
    use_enhancer = any(k.startswith("enhancer.") for k in state_dict)

    return {
        "sr": sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "in_channels": in_channels,
        "dim": dim,
        "n_enc_layers": n_enc_layers,
        "n_dec_layers": n_dec_layers,
        "n_voices": n_voices,
        "mode": mode,
        "legacy_slots": legacy_slots,
        "use_inpaint": use_inpaint,
        "use_enhancer": use_enhancer,
        "arch_version": arch_version,
    }


def _suppress_voice_bleed(
    model: HarmonicSlotTF,
    voice_stems: torch.Tensor,
    active_mask: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Apply inter-voice Wiener competition and frame-dominance gating on CPU float32 [V, C, N]."""
    if strength <= 0.0 or int(active_mask.sum()) < 2:
        return voice_stems
    idx = active_mask.nonzero(as_tuple=True)[0]
    out = voice_stems.clone()
    with torch.no_grad():
        X_v = model._stft(out[idx])  # [V_act, C, F, T]
        mag = X_v.abs().mean(dim=1, keepdim=True)  # [V_act, 1, F, T]
        p = 1.0 + 3.0 * strength
        w_bin = mag.pow(p) / (mag.pow(p).sum(dim=0, keepdim=True) + 1e-8)

        frame_e = mag.pow(2).mean(dim=2, keepdim=True)  # [V_act, 1, 1, T]
        V_act, _, _, T_frames = frame_e.shape
        frame_e = torch.nn.functional.avg_pool1d(
            frame_e.reshape(V_act, 1, T_frames), kernel_size=5, stride=1, padding=2
        ).reshape(V_act, 1, 1, T_frames)
        frame_share = frame_e / (frame_e.sum(dim=0, keepdim=True) + 1e-8)
        thresh = 0.20 * strength
        w_frame = ((frame_share - thresh) / max(1e-4, 0.5 - thresh)).clamp(0.0, 1.0)

        out[idx] = model._istft(X_v * w_bin * w_frame, length=voice_stems.shape[-1])
    return out


def _voice_timbre_print(stems_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized log-spectrum voiceprint [V, F_vp] and energy weight [V] for voiced frames."""
    V, C, N = stems_v.shape
    mono = stems_v.mean(dim=1).float()  # [V, N]
    if mono.shape[-1] < 1024:
        mono = torch.nn.functional.pad(mono, (0, 1024 - mono.shape[-1]))
    win = torch.hann_window(1024, device=mono.device)
    spec = torch.stft(mono, 1024, 256, 1024, win, center=True, return_complex=True).abs()  # [V, 513, T]
    # Focus on speech-band bins (~100 Hz .. ~8 kHz at 48k -> bins 2..170)
    spec_band = spec[:, 2:170, :]
    frame_e = spec_band.pow(2).mean(dim=1)  # [V, T]
    peak_e = frame_e.amax(dim=-1, keepdim=True).clamp_min(1e-10)
    voiced = (frame_e > 0.05 * peak_e).float().unsqueeze(1)  # [V, 1, T]
    log_spec = torch.log1p(spec_band * 100.0)
    mean_print = (log_spec * voiced).sum(dim=-1) / voiced.sum(dim=-1).clamp_min(1.0)
    mean_print = mean_print - mean_print.mean(dim=-1, keepdim=True)
    rms = mono.pow(2).mean(dim=-1).sqrt()
    norm_print = torch.nn.functional.normalize(mean_print, dim=-1) * (rms > 1e-4).float().unsqueeze(-1)
    return norm_print, rms


def _match_voice_permutation(
    prev_tail: torch.Tensor,
    cur_head: torch.Tensor,
    prev_prints: torch.Tensor | None = None,
    cur_prints: torch.Tensor | None = None,
    cur_rms: torch.Tensor | None = None,
) -> Tuple[int, ...]:
    """Match V voice slots across chunks using overlap waveform + running speaker voiceprints."""
    n_voices = prev_tail.shape[0]
    a = prev_tail.reshape(n_voices, -1).float()
    b = cur_head.reshape(n_voices, -1).float()
    a_rms = a.pow(2).mean(dim=-1, keepdim=True).sqrt()
    b_rms = b.pow(2).mean(dim=-1, keepdim=True).sqrt()
    a_norm = torch.nn.functional.normalize(a, dim=-1)
    b_norm = torch.nn.functional.normalize(b, dim=-1)
    # Gate overlap correlation by min overlap energy so silent overlaps don't swap slots
    ov_gate = torch.clamp(torch.minimum(a_rms, b_rms.t()) / 5e-3, max=1.0)
    score_mat = (a_norm @ b_norm.t()) * ov_gate

    if prev_prints is not None and cur_prints is not None and cur_rms is not None:
        vp_sim = torch.nn.functional.normalize(prev_prints, dim=-1) @ torch.nn.functional.normalize(cur_prints, dim=-1).t()
        vp_gate = (prev_prints.norm(dim=-1, keepdim=True) > 0.1).float() * (cur_rms.unsqueeze(0) > 1e-4).float()
        score_mat = score_mat + 0.65 * vp_sim * vp_gate

    best_perm = tuple(range(n_voices))
    best_score = -float("inf")
    for perm in itertools.permutations(range(n_voices)):
        score = sum(float(score_mat[i, perm[i]]) for i in range(n_voices))
        if score > best_score:
            best_score = score
            best_perm = perm
    return best_perm


def _overlap_swap_safety_perm(
    prev_tail: torch.Tensor,
    cur_head: torch.Tensor,
    margin: float = 0.25,
) -> Tuple[int, ...]:
    """Detect clear slot swaps on the overlap window while preserving identity otherwise."""
    n_voices = prev_tail.shape[0]
    a = prev_tail.reshape(n_voices, -1).float()
    b = cur_head.reshape(n_voices, -1).float()
    a_rms = a.pow(2).mean(dim=-1, keepdim=True).sqrt()
    b_rms = b.pow(2).mean(dim=-1, keepdim=True).sqrt()
    a_norm = torch.nn.functional.normalize(a, dim=-1)
    b_norm = torch.nn.functional.normalize(b, dim=-1)
    ov_gate = torch.clamp(torch.minimum(a_rms, b_rms.t()) / 5e-3, max=1.0)
    score_mat = (a_norm @ b_norm.t()) * ov_gate

    id_perm = tuple(range(n_voices))
    id_score = sum(float(score_mat[i, i]) for i in range(n_voices))
    best_perm = id_perm
    best_score = id_score
    for perm in itertools.permutations(range(n_voices)):
        score = sum(float(score_mat[i, perm[i]]) for i in range(n_voices))
        if score > best_score:
            best_score = score
            best_perm = perm
    if best_perm != id_perm and best_score > id_score + margin:
        return best_perm
    return id_perm


def _run_model_chunk(
    model: HarmonicSlotTF,
    chunk_wav: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    activity_threshold: float,
    prev_slots: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run one chunk [1, C_model, N] -> (stems [5, C_model, N], probs [V], slots [1, V, D])."""
    with torch.no_grad():
        x = chunk_wav.to(device=device, dtype=torch.float32)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(device.type == "cuda" and amp_dtype != torch.float32),
        ):
            out = model(x, prev_slots=prev_slots)
        stems = out["stems"][0].float().cpu()  # [S, C_model, N]
        probs = torch.sigmoid(out["activity_logits"][0, : model.n_voices].float()).cpu()
        cur_slots = out["slots"][:, : model.n_voices].detach() if out.get("slots") is not None else None

    if activity_threshold > 0.0:
        keep = (probs >= activity_threshold).to(stems.dtype).view(-1, 1, 1)
        stems[: model.n_voices] = stems[: model.n_voices] * keep
    return stems, probs, cur_slots


def _separate_with_ola(
    model: HarmonicSlotTF,
    wav_bcn: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    chunk_seconds: float,
    overlap_seconds: float,
    activity_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Separate [B, C_model, N] audio -> (stems [B, S, C_model, N], mean_probs [B, V])."""
    B, C_m, N = wav_bcn.shape
    sr = model.sr
    chunk_len = int(round(chunk_seconds * sr)) if chunk_seconds > 0 else N
    overlap_len = min(int(round(overlap_seconds * sr)), max(0, chunk_len // 2))

    if N <= chunk_len or chunk_len <= 0:
        stems_list, probs_list = [], []
        for b in range(B):
            stems, probs, _ = _run_model_chunk(
                model, wav_bcn[b : b + 1], device, amp_dtype, activity_threshold
            )
            stems_list.append(stems)
            probs_list.append(probs)
        return torch.stack(stems_list, dim=0), torch.stack(probs_list, dim=0)

    step = max(1, chunk_len - overlap_len)
    n_sources = model.n_sources
    n_voices = model.n_voices
    is_v3 = getattr(model, "arch_version", "v2") == "v3"
    use_streaming_slots = is_v3 and bool(getattr(model, "trained_streaming", False))
    act_cut = max(activity_threshold, 0.65)
    batch_stems = torch.zeros(B, n_sources, C_m, N, dtype=torch.float32)
    batch_probs = torch.zeros(B, n_voices, dtype=torch.float32)

    fade_in = (
        torch.linspace(0.0, 1.0, overlap_len, dtype=torch.float32).view(1, 1, -1)
        if overlap_len > 0
        else None
    )

    for b in range(B):
        out_acc = torch.zeros(n_sources, C_m, N, dtype=torch.float32)
        prob_acc = torch.zeros(n_voices, dtype=torch.float32)
        running_prints: torch.Tensor | None = None
        carry_slots: torch.Tensor | None = None
        slot_seen = [False] * n_voices
        has_vq = hasattr(model, "slot_module") and hasattr(model.slot_module, "voice_queries")
        weight_sum = 0.0
        pos = 0
        prev_end = 0

        while pos < N:
            end = min(N, pos + chunk_len)
            chunk = wav_bcn[b : b + 1, :, pos:end]
            pad = chunk_len - chunk.shape[-1]
            if pad > 0:
                chunk = torch.nn.functional.pad(chunk, (0, pad))

            used_prev = use_streaming_slots and any(slot_seen) and (carry_slots is not None)
            prev_in = None
            if used_prev and carry_slots is not None:
                prev_in = carry_slots.clone()
                if has_vq:
                    base_q = model.slot_module.voice_queries.to(device=device, dtype=prev_in.dtype)
                    for v in range(n_voices):
                        if not slot_seen[v]:
                            prev_in[:, v] = base_q[:, v]
            stems, probs, cur_slots = _run_model_chunk(
                model, chunk, device, amp_dtype, activity_threshold,
                prev_slots=prev_in,
            )
            valid_len = end - pos
            stems = stems[:, :, :valid_len]
            chunk_rms = float(chunk[:, :, :valid_len].float().pow(2).mean().sqrt())
            cur_prints, cur_rms = _voice_timbre_print(stems[:n_voices])

            ov = max(0, prev_end - pos)
            if pos > 0:
                if used_prev:
                    perm = (
                        _overlap_swap_safety_perm(
                            out_acc[:n_voices, :, pos:prev_end],
                            stems[:n_voices, :, :ov],
                            margin=0.25,
                        )
                        if ov > 0
                        else tuple(range(n_voices))
                    )
                else:
                    prev_ov = out_acc[:n_voices, :, pos:prev_end] if ov > 0 else torch.zeros(n_voices, C_m, 1)
                    cur_ov = stems[:n_voices, :, :ov] if ov > 0 else torch.zeros(n_voices, C_m, 1)
                    perm = _match_voice_permutation(
                        prev_ov,
                        cur_ov,
                        prev_prints=running_prints,
                        cur_prints=cur_prints,
                        cur_rms=cur_rms,
                    )
                if perm != tuple(range(n_voices)):
                    order = list(perm) + list(range(n_voices, n_sources))
                    stems = stems[order]
                    probs = probs[list(perm)]
                    cur_prints = cur_prints[list(perm)]
                    cur_rms = cur_rms[list(perm)]
                    if cur_slots is not None:
                        cur_slots = cur_slots[:, list(perm), :]

                if ov > 0:
                    fi = (
                        fade_in
                        if (fade_in is not None and ov == overlap_len)
                        else torch.linspace(0.0, 1.0, ov).view(1, 1, -1)
                    )
                    fo = 1.0 - fi
                    out_acc[:, :, pos:prev_end] = out_acc[:, :, pos:prev_end] * fo + stems[:, :, :ov] * fi
                    out_acc[:, :, prev_end:end] = stems[:, :, ov:]
                else:
                    out_acc[:, :, pos:end] = stems
            else:
                out_acc[:, :, pos:end] = stems

            if use_streaming_slots and cur_slots is not None:
                if carry_slots is None:
                    if has_vq:
                        carry_slots = (
                            model.slot_module.voice_queries.expand(1, -1, -1)
                            .to(device=cur_slots.device, dtype=cur_slots.dtype)
                            .clone()
                        )
                    else:
                        carry_slots = cur_slots.clone()
                min_slot_rms = max(1e-4, 0.03 * chunk_rms)
                for v in range(n_voices):
                    if float(probs[v]) >= act_cut and float(cur_rms[v]) > min_slot_rms:
                        if not slot_seen[v]:
                            carry_slots[:, v] = cur_slots[:, v]
                            slot_seen[v] = True
                        else:
                            carry_slots[:, v] = 0.85 * carry_slots[:, v] + 0.15 * cur_slots[:, v]

            if not use_streaming_slots:
                if running_prints is None:
                    running_prints = cur_prints * cur_rms.unsqueeze(-1)
                else:
                    running_prints = running_prints + cur_prints * cur_rms.unsqueeze(-1)

            prob_acc = prob_acc + probs * float(valid_len)
            weight_sum += float(valid_len)
            prev_end = end
            if end >= N:
                break
            pos += step

        batch_stems[b] = out_acc
        batch_probs[b] = prob_acc / max(1.0, weight_sum)

    return batch_stems, batch_probs


class HarmonicSlotTFLoader:
    """Load a HarmonicSlotTF .pt or .safetensors checkpoint from ComfyUI/models/audiosep."""

    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list("audiosep")
        return {
            "required": {
                "model_name": (files if files else [_EMPTY_PLACEHOLDER],),
                "precision": (["auto", "bfloat16", "float32"], {"default": "auto"}),
            },
            "optional": {
                "sr_override": ("INT", {"default": 0, "min": 0, "max": 96000, "step": 1000}),
                "hop_length_override": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            },
        }

    RETURN_TYPES = ("HARMONIC_SEP_MODEL", "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load_model"
    CATEGORY = "audio/separation"

    @classmethod
    def IS_CHANGED(cls, model_name: str, **kwargs) -> float:
        if model_name == _EMPTY_PLACEHOLDER:
            return 0.0
        return os.path.getmtime(folder_paths.get_full_path_or_raise("audiosep", model_name))

    def load_model(
        self,
        model_name: str,
        precision: str = "auto",
        sr_override: int = 0,
        hop_length_override: int = 0,
    ):
        if model_name == _EMPTY_PLACEHOLDER:
            raise FileNotFoundError(
                f"No AudioSep checkpoints found. Put .pt or .safetensors files into: {_AUDIOSEP_DIR}"
            )
        ckpt_path = folder_paths.get_full_path_or_raise("audiosep", model_name)
        state_dict, meta = _load_checkpoint_data(ckpt_path)
        if meta.get("model_type") == "CrossStemDebleedRefiner" or "voice_net.in_conv.weight" in state_dict:
            raise ValueError(
                f"'{model_name}' is a CrossStemDebleedRefiner checkpoint. "
                "Load it with 'Craft Crosstalk De-Bleed Loader' instead of 'HarmonicSlotTF Loader'."
            )
        cfg = _infer_model_config(
            state_dict, meta, sr_override=sr_override, hop_override=hop_length_override
        )
        trained_streaming = bool(meta.get("trained_streaming", False))

        model = HarmonicSlotTF(**cfg)
        model.load_state_dict(state_dict, strict=True)
        model.trained_streaming = trained_streaming
        model.eval()

        step = meta.get("step", "?")
        best_sdr = meta.get("best_sdr")
        sdr_str = f"{best_sdr:.2f} dB" if isinstance(best_sdr, (int, float)) else "n/a"
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        stream_str = f" stream={trained_streaming}" if cfg["arch_version"] == "v3" else ""
        info = (
            f"{model_name} | arch={cfg['arch_version']}{stream_str} | mode={cfg['mode']} | "
            f"{cfg['sr']}Hz {cfg['in_channels']}ch | "
            f"n_fft={cfg['n_fft']} hop={cfg['hop_length']} dim={cfg['dim']} | "
            f"{n_params:.2f}M params | step={step} best_sdr={sdr_str}"
        )
        bundle = {
            "model": model,
            "config": cfg,
            "precision": precision,
            "trained_streaming": trained_streaming,
            "info": info,
        }
        return (bundle, info)


class HarmonicSlotTFSeparate:
    """Separate audio into Voice 1, Voice 2, Voice 3, All Voices, Music, and Ambience."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HARMONIC_SEP_MODEL",),
                "audio": ("AUDIO",),
                "chunk_seconds": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "max": 120.0, "step": 0.5},
                ),
                "overlap_seconds": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "activity_threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05},
                ),
                "voice_bleed_suppression": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "match_input_sample_rate": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = (
        "voice_1",
        "voice_2",
        "voice_3",
        "all_voices",
        "music",
        "ambience",
        "activity_info",
    )
    FUNCTION = "separate"
    CATEGORY = "audio/separation"

    def separate(
        self,
        model: Dict[str, Any],
        audio: Dict[str, Any],
        chunk_seconds: float = 6.0,
        overlap_seconds: float = 1.5,
        activity_threshold: float = 0.0,
        voice_bleed_suppression: float = 0.0,
        match_input_sample_rate: bool = False,
    ):
        net: HarmonicSlotTF = model["model"]
        precision: str = model.get("precision", "auto")

        wav = audio["waveform"].detach().cpu().float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.ndim == 2:
            wav = wav.unsqueeze(0)

        in_sr = int(audio["sample_rate"])
        orig_ch = wav.shape[1]
        orig_len = wav.shape[-1]

        if in_sr != net.sr:
            wav_model = torchaudio.functional.resample(wav, in_sr, net.sr)
        else:
            wav_model = wav

        if net.in_channels == 1 and wav_model.shape[1] > 1:
            wav_model = wav_model.mean(dim=1, keepdim=True)
        elif net.in_channels == 2 and wav_model.shape[1] == 1:
            wav_model = wav_model.expand(-1, 2, -1).contiguous()

        device = comfy.model_management.get_torch_device()
        if precision == "float32" or device.type != "cuda":
            amp_dtype = torch.float32
        elif precision == "bfloat16" or torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float32

        net.to(device)
        try:
            stems, probs = _separate_with_ola(
                net,
                wav_model,
                device=device,
                amp_dtype=amp_dtype,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
                activity_threshold=activity_threshold,
            )
        finally:
            net.to("cpu")
            comfy.model_management.soft_empty_cache()

        raw_v_all = stems[:, : net.n_voices].sum(dim=1, keepdim=True)
        if voice_bleed_suppression > 0.0:
            min_p = activity_threshold if activity_threshold > 0.0 else 0.35
            for b in range(stems.shape[0]):
                v_chunk = stems[b, : net.n_voices]
                active_mask = (probs[b] >= min_p) & (v_chunk.abs().amax(dim=(1, 2)) > 1e-5)
                stems[b, : net.n_voices] = _suppress_voice_bleed(
                    net, v_chunk, active_mask, voice_bleed_suppression
                )

        stems_all = torch.cat([stems, raw_v_all], dim=1)
        out_sr = net.sr
        if match_input_sample_rate and out_sr != in_sr:
            B, S_all, C_m, T_m = stems_all.shape
            stems_all = torchaudio.functional.resample(
                stems_all.reshape(B * S_all, C_m, T_m), out_sr, in_sr
            )
            if stems_all.shape[-1] > orig_len:
                stems_all = stems_all[..., :orig_len]
            elif stems_all.shape[-1] < orig_len:
                stems_all = torch.nn.functional.pad(
                    stems_all, (0, orig_len - stems_all.shape[-1])
                )
            stems_all = stems_all.reshape(B, S_all, C_m, orig_len)
            out_sr = in_sr

        if orig_ch == 1 and stems_all.shape[2] > 1:
            stems_all = stems_all.mean(dim=2, keepdim=True)
        elif orig_ch == 2 and stems_all.shape[2] == 1:
            stems_all = stems_all.expand(-1, -1, 2, -1).contiguous()

        v1 = {"waveform": stems_all[:, 0].contiguous(), "sample_rate": out_sr}
        v2 = {"waveform": stems_all[:, 1].contiguous(), "sample_rate": out_sr}
        v3 = {"waveform": stems_all[:, 2].contiguous(), "sample_rate": out_sr}
        v_all = {"waveform": stems_all[:, -1].contiguous(), "sample_rate": out_sr}
        music = {"waveform": stems_all[:, 3].contiguous(), "sample_rate": out_sr}
        ambience = {"waveform": stems_all[:, 4].contiguous(), "sample_rate": out_sr}

        mean_p = probs.mean(dim=0)
        act_str = ", ".join(f"V{i + 1}={float(mean_p[i]):.2f}" for i in range(net.n_voices))
        info = f"{model['info']} | output={out_sr}Hz {stems_all.shape[2]}ch | active: {act_str}"
        return (v1, v2, v3, v_all, music, ambience, info)


def _infer_refiner_config(
    state_dict: Dict[str, torch.Tensor],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Infer CrossStemDebleedRefiner constructor kwargs from metadata and tensor shapes."""
    if meta.get("model_type") != "CrossStemDebleedRefiner" and "voice_net.in_conv.weight" not in state_dict:
        raise ValueError(
            "Selected checkpoint is not a CrossStemDebleedRefiner. "
            "Use 'HarmonicSlotTF Loader' for main separation models."
        )
    out_ch = int(state_dict["voice_net.out_conv.weight"].shape[0])
    in_channels = int(meta.get("in_channels") or (out_ch // 13))
    voice_base_c = int(meta.get("voice_base_c") or state_dict["voice_net.in_conv.weight"].shape[0])
    bg_base_c = int(meta.get("bg_base_c") or state_dict["bg_net.in_conv.weight"].shape[0])
    v_bn_idx = [
        int(k.split(".")[2])
        for k in state_dict
        if k.startswith("voice_net.bottleneck.") and k.split(".")[2].isdigit()
    ]
    b_bn_idx = [
        int(k.split(".")[2])
        for k in state_dict
        if k.startswith("bg_net.bottleneck.") and k.split(".")[2].isdigit()
    ]
    voice_bottleneck = int(meta.get("voice_bottleneck") or (max(v_bn_idx) + 1 if v_bn_idx else 2))
    bg_bottleneck = int(meta.get("bg_bottleneck") or (max(b_bn_idx) + 1 if b_bn_idx else 1))
    wave_hidden = int(
        meta.get("wave_hidden")
        or (state_dict["voice_wave_net.0.weight"].shape[0] if "voice_wave_net.0.weight" in state_dict else 24)
    )
    return {
        "sr": int(meta.get("sr", 48000)),
        "n_fft": int(meta.get("n_fft", 2048)),
        "hop_length": int(meta.get("hop_length", 256)),
        "in_channels": in_channels,
        "n_voices": int(meta.get("n_voices", 3)),
        "voice_base_c": voice_base_c,
        "bg_base_c": bg_base_c,
        "voice_bottleneck": voice_bottleneck,
        "bg_bottleneck": bg_bottleneck,
        "wave_hidden": wave_hidden,
    }


def _prep_audio_tensor(
    audio: Dict[str, Any], target_sr: int, target_ch: int, target_len: int | None = None
) -> Tuple[torch.Tensor, int, int]:
    """Normalize ComfyUI AUDIO dict to float32 [B, target_ch, N] at target_sr."""
    wav = audio["waveform"].detach().cpu().float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0).unsqueeze(0)
    elif wav.ndim == 2:
        wav = wav.unsqueeze(0)
    in_sr = int(audio["sample_rate"])
    orig_ch = wav.shape[1]
    if in_sr != target_sr:
        wav = torchaudio.functional.resample(wav, in_sr, target_sr)
    if target_ch == 1 and wav.shape[1] > 1:
        wav = wav.mean(dim=1, keepdim=True)
    elif target_ch == 2 and wav.shape[1] == 1:
        wav = wav.expand(-1, 2, -1).contiguous()
    if target_len is not None:
        if wav.shape[-1] > target_len:
            wav = wav[..., :target_len]
        elif wav.shape[-1] < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[-1]))
    return wav, in_sr, orig_ch


@torch.no_grad()
def _refine_with_ola(
    refiner: CrossStemDebleedRefiner,
    raw_stems: torch.Tensor,
    mix: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    chunk_seconds: float,
    overlap_seconds: float,
) -> torch.Tensor:
    """Run CrossStemDebleedRefiner over arbitrary length with raised-cosine OLA (strict slot order)."""
    B, S, C, N = raw_stems.shape
    use_amp = device.type == "cuda" and amp_dtype != torch.float32
    chunk_len = int(round(chunk_seconds * refiner.sr)) if chunk_seconds > 0 else N
    chunk_len = max(refiner.n_fft * 2, min(chunk_len, N))

    if chunk_len >= N:
        out_b = torch.zeros_like(raw_stems)
        for b in range(B):
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out_b[b] = refiner(raw_stems[b : b + 1].to(device), mix[b : b + 1].to(device))["stems"][0].float().cpu()
        return out_b

    overlap = max(0, min(int(round(overlap_seconds * refiner.sr)), chunk_len // 2))
    step = max(refiner.hop_length, chunk_len - overlap)
    out_all = torch.zeros_like(raw_stems)

    for b in range(B):
        acc = torch.zeros(S, C, N, dtype=torch.float32)
        w_acc = torch.zeros(1, 1, N, dtype=torch.float32)
        pos = 0
        while pos < N:
            end = min(N, pos + chunk_len)
            start = max(0, end - chunk_len) if (end - pos) < chunk_len else pos
            c_raw = raw_stems[b : b + 1, :, :, start:end].to(device)
            c_mix = mix[b : b + 1, :, start:end].to(device)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                c_ref = refiner(c_raw, c_mix)["stems"][0].float().cpu()

            clen = end - start
            win = torch.ones(clen, dtype=torch.float32)
            fade = min(overlap, clen // 2)
            if fade > 1:
                ramp = 0.5 * (1.0 - torch.cos(torch.linspace(0.0, 3.141592653589793, fade)))
                if start > 0:
                    win[:fade] = ramp
                if end < N:
                    win[-fade:] = ramp.flip(0)
            w_view = win.view(1, 1, clen)
            acc[:, :, start:end] += c_ref * w_view
            w_acc[:, :, start:end] += w_view

            if end >= N:
                break
            pos += step
        out_all[b] = acc / w_acc.clamp_min(1e-6)
    return out_all


class CraftDebleedRefinerLoader:
    """Load a CrossStemDebleedRefiner (.safetensors or .pt) from ComfyUI/models/audiosep."""

    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list("audiosep")
        return {
            "required": {
                "model_name": (files if files else [_EMPTY_PLACEHOLDER],),
                "precision": (["auto", "bfloat16", "float32"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("CRAFT_REFINER_MODEL", "STRING")
    RETURN_NAMES = ("refiner", "refiner_info")
    FUNCTION = "load_refiner"
    CATEGORY = "audio/separation"

    @classmethod
    def IS_CHANGED(cls, model_name: str, **kwargs) -> float:
        if model_name == _EMPTY_PLACEHOLDER:
            return 0.0
        return os.path.getmtime(folder_paths.get_full_path_or_raise("audiosep", model_name))

    def load_refiner(self, model_name: str, precision: str = "auto"):
        if model_name == _EMPTY_PLACEHOLDER:
            raise FileNotFoundError(
                f"No AudioSep checkpoints found. Put .pt or .safetensors files into: {_AUDIOSEP_DIR}"
            )
        ckpt_path = folder_paths.get_full_path_or_raise("audiosep", model_name)
        state_dict, meta = _load_checkpoint_data(ckpt_path)
        cfg = _infer_refiner_config(state_dict, meta)

        refiner = CrossStemDebleedRefiner(**cfg)
        refiner.load_state_dict(state_dict, strict=True)
        refiner.eval()

        step = meta.get("step", "?")
        sir = meta.get("sir_bleed")
        sir_str = f"{sir:.1f} dB" if isinstance(sir, (int, float)) else "n/a"
        ver = meta.get("refiner_version", "v2")
        n_params = sum(p.numel() for p in refiner.parameters()) / 1e6
        info = (
            f"{model_name} | {ver} | {cfg['sr']}Hz {cfg['in_channels']}ch | "
            f"n_fft={cfg['n_fft']} hop={cfg['hop_length']} | {n_params:.2f}M params | "
            f"step={step} sir={sir_str}"
        )
        bundle = {"model": refiner, "config": cfg, "precision": precision, "info": info}
        return (bundle, info)


class CraftDebleedRefinerApply:
    """Chain after HarmonicSlotTF Separate (5-Stem) to scrub crosstalk, pause bleed, and STFT grain."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refiner": ("CRAFT_REFINER_MODEL",),
                "mix_audio": ("AUDIO",),
                "voice_1": ("AUDIO",),
                "voice_2": ("AUDIO",),
                "voice_3": ("AUDIO",),
                "music": ("AUDIO",),
                "ambience": ("AUDIO",),
                "chunk_seconds": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 60.0, "step": 0.5}),
                "overlap_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = (
        "voice_1",
        "voice_2",
        "voice_3",
        "all_voices",
        "music",
        "ambience",
        "refiner_info",
    )
    FUNCTION = "refine"
    CATEGORY = "audio/separation"

    def refine(
        self,
        refiner: Dict[str, Any],
        mix_audio: Dict[str, Any],
        voice_1: Dict[str, Any],
        voice_2: Dict[str, Any],
        voice_3: Dict[str, Any],
        music: Dict[str, Any],
        ambience: Dict[str, Any],
        chunk_seconds: float = 6.0,
        overlap_seconds: float = 1.0,
        strength: float = 1.0,
    ):
        net: CrossStemDebleedRefiner = refiner["model"]
        precision: str = refiner.get("precision", "auto")

        v1_w, stem_sr, stem_ch = _prep_audio_tensor(voice_1, net.sr, net.in_channels)
        orig_len_in = int(voice_1["waveform"].shape[-1])
        T_m = v1_w.shape[-1]
        v2_w, _, _ = _prep_audio_tensor(voice_2, net.sr, net.in_channels, target_len=T_m)
        v3_w, _, _ = _prep_audio_tensor(voice_3, net.sr, net.in_channels, target_len=T_m)
        mus_w, _, _ = _prep_audio_tensor(music, net.sr, net.in_channels, target_len=T_m)
        amb_w, _, _ = _prep_audio_tensor(ambience, net.sr, net.in_channels, target_len=T_m)
        mix_w, _, _ = _prep_audio_tensor(mix_audio, net.sr, net.in_channels, target_len=T_m)

        raw_stems = torch.stack([v1_w, v2_w, v3_w, mus_w, amb_w], dim=1)  # [B, 5, C, T_m]

        if strength <= 0.0:
            refined = raw_stems
        else:
            device = comfy.model_management.get_torch_device()
            if precision == "float32" or device.type != "cuda":
                amp_dtype = torch.float32
            elif precision == "bfloat16" or torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float32

            net.to(device)
            try:
                refined = _refine_with_ola(
                    net,
                    raw_stems,
                    mix_w,
                    device=device,
                    amp_dtype=amp_dtype,
                    chunk_seconds=chunk_seconds,
                    overlap_seconds=overlap_seconds,
                )
            finally:
                net.to("cpu")
                comfy.model_management.soft_empty_cache()

            if strength < 1.0:
                refined = raw_stems + float(strength) * (refined - raw_stems)

        v_all_w = refined[:, : net.n_voices].sum(dim=1, keepdim=True)
        stems_all = torch.cat([refined, v_all_w], dim=1)  # [B, 6, C, T_m]

        out_sr = net.sr
        if stem_sr != out_sr:
            B, S_all, C_m, L_m = stems_all.shape
            stems_all = torchaudio.functional.resample(
                stems_all.reshape(B * S_all, C_m, L_m), out_sr, stem_sr
            )
            if stems_all.shape[-1] > orig_len_in:
                stems_all = stems_all[..., :orig_len_in]
            elif stems_all.shape[-1] < orig_len_in:
                stems_all = torch.nn.functional.pad(
                    stems_all, (0, orig_len_in - stems_all.shape[-1])
                )
            stems_all = stems_all.reshape(B, S_all, C_m, orig_len_in)
            out_sr = stem_sr

        if stem_ch == 1 and stems_all.shape[2] > 1:
            stems_all = stems_all.mean(dim=2, keepdim=True)
        elif stem_ch == 2 and stems_all.shape[2] == 1:
            stems_all = stems_all.expand(-1, -1, 2, -1).contiguous()

        o_v1 = {"waveform": stems_all[:, 0].contiguous(), "sample_rate": out_sr}
        o_v2 = {"waveform": stems_all[:, 1].contiguous(), "sample_rate": out_sr}
        o_v3 = {"waveform": stems_all[:, 2].contiguous(), "sample_rate": out_sr}
        o_va = {"waveform": stems_all[:, -1].contiguous(), "sample_rate": out_sr}
        o_mu = {"waveform": stems_all[:, 3].contiguous(), "sample_rate": out_sr}
        o_am = {"waveform": stems_all[:, 4].contiguous(), "sample_rate": out_sr}

        info = f"{refiner['info']} | strength={strength:.2f} | output={out_sr}Hz {stems_all.shape[2]}ch"
        return (o_v1, o_v2, o_v3, o_va, o_mu, o_am, info)


NODE_CLASS_MAPPINGS = {
    "HarmonicSlotTFLoader": HarmonicSlotTFLoader,
    "HarmonicSlotTFSeparate": HarmonicSlotTFSeparate,
    "CraftDebleedRefinerLoader": CraftDebleedRefinerLoader,
    "CraftDebleedRefinerApply": CraftDebleedRefinerApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HarmonicSlotTFLoader": "HarmonicSlotTF Loader",
    "HarmonicSlotTFSeparate": "HarmonicSlotTF Separate (5-Stem)",
    "CraftDebleedRefinerLoader": "Craft Crosstalk De-Bleed Loader",
    "CraftDebleedRefinerApply": "Craft Crosstalk De-Bleed Refiner (5-Stem)",
}
