"""Automated dataset downloader & cleaner for HarmonicSlotTF.

Downloads verified public datasets directly into clean folders compatible with train.py:
  out/
    speech/<speaker_id>/...   (LibriSpeech dev-clean or train-clean-100)
    music/*.wav               (100% instrumental: BabySlakh mix.wav &/or MUSDB18-HQ drums+bass+other)
    amb/*.wav                 (DEMAND diverse acoustic environments, ch01.wav)
    rir/*.wav                 (OpenSLR 28 simulated Room Impulse Responses)

Presets:
  quick        (~1.8 GB):  LibriSpeech dev-clean + BabySlakh + 6 DEMAND 16k envs
  ablation_16k (~10.2 GB): LibriSpeech train-clean-100 + BabySlakh + all 17 DEMAND 16k + RIRs
  full_48k     (~6 GB):    5-language 48 kHz single-speaker speech (EN/ZH/TA/TE/HI) + Music + DEMAND 48k + RIRs
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

URLS = {
    "dev_clean": "https://openslr.elda.org/resources/12/dev-clean.tar.gz",
    "train_clean_100": "https://openslr.elda.org/resources/12/train-clean-100.tar.gz",
    "babyslakh": "https://zenodo.org/api/records/4603870/files/babyslakh_16k.tar.gz/content",
    "musdb18hq": "https://zenodo.org/api/records/3338373/files/musdb18hq.zip/content",
    "rirs_noises": "https://openslr.elda.org/resources/28/rirs_noises.zip",
}

MULTILINGUAL_SPEECH_48K: list[tuple[str, str]] = [
    ("en_mf", "https://openslr.elda.org/resources/83/midlands_english_female.zip"),
    ("en_mm", "https://openslr.elda.org/resources/83/midlands_english_male.zip"),
    ("en_scf", "https://openslr.elda.org/resources/83/scottish_english_female.zip"),
    ("en_scm", "https://openslr.elda.org/resources/83/scottish_english_male.zip"),
    ("ta_f", "https://openslr.elda.org/resources/65/ta_in_female.zip"),
    ("ta_m", "https://openslr.elda.org/resources/65/ta_in_male.zip"),
    ("te_f", "https://openslr.elda.org/resources/66/te_in_female.zip"),
    ("te_m", "https://openslr.elda.org/resources/66/te_in_male.zip"),
    ("zh_md", "https://openslr.elda.org/resources/68/dev_set.tar.gz"),
    ("hi_in", "https://openslr.elda.org/resources/103/Hindi_test.tar.gz"),
]

DEMAND_QUICK_16K = [
    "DKITCHEN_16k.zip",
    "NFIELD_16k.zip",
    "OHALLWAY_16k.zip",
    "PSTATION_16k.zip",
    "TMETRO_16k.zip",
]

DEMAND_QUICK_48K = [name.replace("_16k.zip", "_48k.zip") for name in DEMAND_QUICK_16K]

DEMAND_ALL_16K = [
    "DKITCHEN_16k.zip", "DWASHING_16k.zip",
    "NFIELD_16k.zip", "NPARK_16k.zip", "NRIVER_16k.zip",
    "OHALLWAY_16k.zip", "OOFFICE_16k.zip",
    "PSTATION_16k.zip", "SPSQUARE_16k.zip", "STRAFFIC_16k.zip",
    "TBUS_16k.zip", "TCAR_16k.zip", "TMETRO_16k.zip",
]

DEMAND_ALL_48K = [name.replace("_16k.zip", "_48k.zip") for name in DEMAND_ALL_16K]

# DEMAND environments verified free of background speech, PA chatter, and radio music
DEMAND_CLEAN_48K = [
    "DKITCHEN_48k.zip", "DWASHING_48k.zip",
    "NFIELD_48k.zip", "NRIVER_48k.zip",
    "OHALLWAY_48k.zip", "OOFFICE_48k.zip",
    "STRAFFIC_48k.zip", "TCAR_48k.zip",
]


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using cached {dest.name}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")

    curl_bin = shutil.which("curl") or shutil.which("curl.exe")
    if curl_bin is not None:
        print(f"Downloading {dest.name} via curl...")
        subprocess.run(
            [curl_bin, "-L", "-C", "-", "-f", "--retry", "3", "-o", str(tmp), url],
            check=True,
        )
        tmp.replace(dest)
        return dest

    existing = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    if existing > 0:
        req.add_header("Range", f"bytes={existing}-")
    print(f"Downloading {url} -> {dest.name} (resume={existing})")
    with urllib.request.urlopen(req, timeout=60) as resp:
        mode = "ab" if existing > 0 and resp.status == 206 else "wb"
        total = resp.headers.get("Content-Length")
        total_bytes = (int(total) + (existing if mode == "ab" else 0)) if total else None
        done = existing if mode == "ab" else 0
        next_report = 50 * 1024 * 1024
        with open(tmp, mode) as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    if total_bytes:
                        print(f"  {dest.name}: {done/1e6:.0f}/{total_bytes/1e6:.0f} MB ({100*done/total_bytes:.0f}%)")
                    else:
                        print(f"  {dest.name}: {done/1e6:.0f} MB")
                    next_report += 50 * 1024 * 1024
    tmp.replace(dest)
    return dest


def _safe_extract_tar(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        try:
            tf.extractall(out_dir, filter="data")
        except TypeError:
            tf.extractall(out_dir)


def _safe_extract_zip(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(out_dir)


def prepare_librispeech(url: str, subset_name: str, out_speech: Path, cache_dir: Path, keep_archives: bool) -> None:
    if out_speech.exists() and any(out_speech.iterdir()):
        print(f"Speech dir {out_speech} already populated, skipping.")
        return
    tgz = download_file(url, cache_dir / f"{subset_name}.tar.gz")
    tmp_dir = cache_dir / f"_extract_{subset_name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    print(f"Extracting {tgz.name}...")
    _safe_extract_tar(tgz, tmp_dir)
    src = tmp_dir / "LibriSpeech" / subset_name
    out_speech.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(out_speech))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if not keep_archives and tgz.exists():
        tgz.unlink()


def prepare_multilingual_48k(out_speech: Path, cache_dir: Path, keep_archives: bool) -> None:
    """Download and group clean single-speaker speech across English, Tamil, Telugu, Chinese, and Hindi."""
    out_speech.mkdir(parents=True, exist_ok=True)
    generic_dirs = {"wav", "wavs", "audio", "dev", "dev_set", "test", "train", "hindi_test", "speech"}
    for tag, url in MULTILINGUAL_SPEECH_48K:
        tmp_dir = cache_dir / f"_extract_{tag}"
        if tmp_dir.exists():
            print(f"Found incomplete extraction for {tag}, cleaning partial files...", flush=True)
            for partial in out_speech.glob(f"{tag}_*"):
                shutil.rmtree(partial, ignore_errors=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if any(out_speech.glob(f"{tag}_*")):
            print(f"Multilingual subset {tag} already populated, skipping.")
            continue
        ext = ".zip" if url.endswith(".zip") else ".tar.gz"
        archive_path = download_file(url, cache_dir / f"{tag}_48k{ext}")
        print(f"Extracting {archive_path.name}...")
        if ext == ".zip":
            _safe_extract_zip(archive_path, tmp_dir)
        else:
            _safe_extract_tar(archive_path, tmp_dir)

        utt_to_spk: dict[str, str] = {}
        for u2s in tmp_dir.rglob("utt2spk"):
            if u2s.is_file():
                for line in u2s.read_text(encoding="utf-8", errors="ignore").splitlines():
                    toks = line.strip().split()
                    if len(toks) >= 2:
                        utt_to_spk[toks[0]] = toks[1]

        count = 0
        skipped_excess = 0
        spk_counts: dict[str, int] = {}
        for wav_path in sorted(tmp_dir.rglob("*")):
            if not wav_path.is_file() or wav_path.suffix.lower() not in {".wav", ".flac"}:
                continue
            parent = wav_path.parent.name
            if wav_path.stem in utt_to_spk:
                spk_tok = utt_to_spk[wav_path.stem]
            elif tag == "zh_md" and parent.lower() not in generic_dirs and parent != tmp_dir.name:
                spk_tok = parent
            elif "-" in wav_path.stem and tag == "hi_in":
                spk_tok = wav_path.stem.split("-")[0]
            else:
                parts = wav_path.stem.split("_")
                if len(parts) >= 3:
                    spk_tok = "_".join(parts[:2])
                elif len(parts) == 2:
                    spk_tok = parts[0]
                else:
                    spk_tok = parent
            spk_id = f"{tag}_{spk_tok}"
            dst_dir = out_speech / spk_id
            if spk_id not in spk_counts:
                spk_counts[spk_id] = len(list(dst_dir.glob("*"))) if dst_dir.exists() else 0
            if spk_counts[spk_id] >= 45:
                wav_path.unlink()
                skipped_excess += 1
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(wav_path), str(dst_dir / wav_path.name))
            spk_counts[spk_id] += 1
            count += 1
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Prepared {count} files for subset {tag} (skipped {skipped_excess} excess raw files).", flush=True)
        if not keep_archives and archive_path.exists():
            archive_path.unlink()


def _speech_frame_stats(
    path: Path, sf_mod, np_mod
) -> tuple[float, float, float, float, float, float] | None:
    """Return (duration_s, peak_abs, p10_floor_db, p98_speech_db, oct_std, med_f0) or None if invalid/short."""
    try:
        x, sr = sf_mod.read(str(path), dtype="float32")
    except Exception:
        return None
    if sr <= 0 or x.size == 0:
        return None
    if x.ndim > 1:
        x = x.mean(axis=1)
    dur = len(x) / float(sr)
    f20 = max(1, int(0.02 * sr))
    n20 = len(x) // f20
    if n20 < 10:
        return None
    frames20 = x[: n20 * f20].reshape(n20, f20)
    db20 = 10.0 * np_mod.log10(np_mod.mean(frames20 ** 2, axis=1) + 1e-12)
    peak_abs = float(np_mod.max(np_mod.abs(x)))
    p10 = float(np_mod.percentile(db20, 10))
    p98 = float(np_mod.percentile(db20, 98))

    fl = max(16, int(0.04 * sr))
    hop = max(8, int(0.02 * sr))
    if len(x) < fl:
        return None
    frames40 = np_mod.lib.stride_tricks.sliding_window_view(x, fl)[::hop]
    db40 = 10.0 * np_mod.log10(np_mod.mean(frames40 ** 2, axis=1) + 1e-12)
    act = frames40[db40 > (float(np_mod.percentile(db40, 98)) - 18.0)]
    if len(act) == 0:
        return None
    act = act - np_mod.mean(act, axis=1, keepdims=True)
    spec = np_mod.fft.rfft(act, n=2 * fl, axis=1)
    ac = np_mod.fft.irfft(spec * np_mod.conj(spec), n=2 * fl, axis=1)[:, :fl]
    ac = ac / (ac[:, :1] + 1e-12)
    lmin = max(1, int(sr / 380))
    lmax = min(fl - 1, int(sr / 75))
    if lmax <= lmin:
        return None
    sub = ac[:, lmin:lmax]
    rmax = np_mod.max(sub, axis=1)
    lags = lmin + np_mod.argmax(sub, axis=1)
    vf = (sr / lags.astype(np_mod.float32))[rmax > 0.65]
    if len(vf) < 3:
        return None
    oct_std = float(np_mod.std(np_mod.log2(vf)))
    med_f0 = float(np_mod.median(vf))
    return dur, peak_abs, p10, p98, oct_std, med_f0


def purify_speech(
    out_speech: Path,
    min_snr_db: float = 42.0,
    max_floor_db: float = -58.0,
    min_peak_db: float = -28.0,
    min_dur_sec: float = 2.0,
    min_files_per_spk: int = 10,
    max_oct_std: float = 0.55,
    max_spk_pitch_dev: float = 0.40,
    max_files_per_spk: int | None = None,
    max_spks_per_lang: int | None = None,
) -> None:
    """Filter out noisy or multi-speaker files/folders so each clip and speaker folder is 1 clean speaker."""
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore

    if not out_speech.exists():
        return

    removed_non_audio = 0
    for p in list(out_speech.rglob("*")):
        if p.is_file() and p.suffix.lower() not in {".wav", ".flac"}:
            p.unlink()
            removed_non_audio += 1

    kept_spks = 0
    kept_files = 0
    removed_spks = 0
    removed_files = 0
    all_floors: list[float] = []
    all_snrs: list[float] = []
    surviving_spks: list[tuple[Path, float, list[float], list[float]]] = []

    print(
        f"Purifying speech under {out_speech} "
        f"(p10<={max_floor_db}dB, SNR>={min_snr_db}dB, p98>={min_peak_db}dB, "
        f"dur>={min_dur_sec}s, oct_std<={max_oct_std}, spk_dev<={max_spk_pitch_dev}, "
        f"max_files_per_spk={max_files_per_spk}, max_spks_per_lang={max_spks_per_lang})...",
        flush=True,
    )

    for spk_dir in [d for d in sorted(out_speech.iterdir()) if d.is_dir()]:
        audio_files = [
            p for p in sorted(spk_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in {".wav", ".flac"}
        ]
        if max_files_per_spk is not None and len(audio_files) > max_files_per_spk * 2:
            keep_limit = max_files_per_spk * 2
            for extra_p in audio_files[keep_limit:]:
                extra_p.unlink()
                removed_files += 1
            audio_files = audio_files[:keep_limit]

        if len(audio_files) < min_files_per_spk:
            shutil.rmtree(spk_dir, ignore_errors=True)
            removed_spks += 1
            removed_files += len(audio_files)
            continue

        # Fast speaker pre-screen on up to 6 utterances
        sample_floors: list[float] = []
        sample_snrs: list[float] = []
        for sample_p in audio_files[:6]:
            stats = _speech_frame_stats(sample_p, sf, np)
            if stats is not None:
                _, _, p10, p98, _, _ = stats
                sample_floors.append(p10)
                sample_snrs.append(p98 - p10)

        if (
            not sample_floors
            or float(np.median(sample_floors)) > max_floor_db + 3.0
            or float(np.median(sample_snrs)) < min_snr_db - 3.0
        ):
            shutil.rmtree(spk_dir, ignore_errors=True)
            removed_spks += 1
            removed_files += len(audio_files)
            continue

        # Layer 1 (SNR/floor) + Layer 2 (intra-file single-speaker pitch stability)
        candidates: list[tuple[Path, float, float, float]] = []
        for p in audio_files:
            stats = _speech_frame_stats(p, sf, np)
            if stats is None:
                p.unlink()
                removed_files += 1
                continue
            dur, peak_abs, p10, p98, oct_std, med_f0 = stats
            snr = p98 - p10
            if (
                dur < min_dur_sec
                or peak_abs >= 0.999
                or p10 > max_floor_db
                or p98 < min_peak_db
                or snr < min_snr_db
                or oct_std > max_oct_std
            ):
                p.unlink()
                removed_files += 1
            else:
                candidates.append((p, p10, snr, med_f0))

        # Layer 3 (per-speaker folder pitch-register consistency)
        surviving: list[tuple[Path, float, float]] = []
        if candidates:
            spk_med_f0 = float(np.median([c[3] for c in candidates]))
            for p, p10, snr, med_f0 in candidates:
                if abs(math.log2(max(med_f0, 1.0) / max(spk_med_f0, 1.0))) > max_spk_pitch_dev:
                    p.unlink()
                    removed_files += 1
                else:
                    surviving.append((p, p10, snr))

        if max_files_per_spk is not None and len(surviving) > max_files_per_spk:
            surviving.sort(key=lambda x: x[2], reverse=True)
            for p, _, _ in surviving[max_files_per_spk:]:
                p.unlink()
                removed_files += 1
            surviving = surviving[:max_files_per_spk]

        for sub in sorted(spk_dir.rglob("*"), reverse=True):
            if sub.is_dir() and not any(sub.iterdir()):
                sub.rmdir()

        if len(surviving) < min_files_per_spk:
            shutil.rmtree(spk_dir, ignore_errors=True)
            removed_spks += 1
            removed_files += len(surviving)
        else:
            spk_floors = [x[1] for x in surviving]
            spk_snrs = [x[2] for x in surviving]
            surviving_spks.append((spk_dir, float(np.median(spk_snrs)), spk_floors, spk_snrs))

    if max_spks_per_lang is not None and surviving_spks:
        by_lang: dict[str, list[tuple[Path, float, list[float], list[float]]]] = {}
        for item in surviving_spks:
            lang = item[0].name.split("_")[0]
            by_lang.setdefault(lang, []).append(item)
        final_spks: list[tuple[Path, float, list[float], list[float]]] = []
        for lang in sorted(by_lang):
            items = by_lang[lang]
            if len(items) > max_spks_per_lang:
                items.sort(key=lambda x: x[1], reverse=True)
                for spk_dir, _, floors, _ in items[max_spks_per_lang:]:
                    shutil.rmtree(spk_dir, ignore_errors=True)
                    removed_spks += 1
                    removed_files += len(floors)
                items = items[:max_spks_per_lang]
            final_spks.extend(items)
        surviving_spks = final_spks

    for _, _, spk_floors, spk_snrs in surviving_spks:
        kept_spks += 1
        kept_files += len(spk_floors)
        all_floors.extend(spk_floors)
        all_snrs.extend(spk_snrs)

    if all_floors:
        worst_floor = max(all_floors)
        med_floor = float(np.median(all_floors))
        min_snr = min(all_snrs)
        med_snr = float(np.median(all_snrs))
        stats_msg = (
            f"floor(p10) worst={worst_floor:.1f}dB median={med_floor:.1f}dB | "
            f"SNR(p98-p10) min={min_snr:.1f}dB median={med_snr:.1f}dB"
        )
    else:
        stats_msg = "no speech files survived"

    print(
        f"Purify complete: kept {kept_spks} speakers / {kept_files} files | "
        f"removed {removed_spks} noisy speakers, {removed_files} files, {removed_non_audio} non-audio files | "
        f"{stats_msg}",
        flush=True,
    )


def prepare_babyslakh(out_music: Path, cache_dir: Path, keep_archives: bool) -> None:
    out_music.mkdir(parents=True, exist_ok=True)
    if list(out_music.glob("babyslakh_*.wav")):
        print("BabySlakh music already prepared, skipping.")
        return
    tgz = download_file(URLS["babyslakh"], cache_dir / "babyslakh_16k.tar.gz")
    tmp_dir = cache_dir / "_extract_babyslakh"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    print("Extracting BabySlakh (keeping instrumental mix.wav tracks)...")
    _safe_extract_tar(tgz, tmp_dir)
    count = 0
    for mix_wav in sorted(tmp_dir.rglob("mix.wav")):
        track_id = mix_wav.parent.name
        shutil.move(str(mix_wav), str(out_music / f"babyslakh_{track_id}.wav"))
        count += 1
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Prepared {count} BabySlakh instrumental tracks.")
    if not keep_archives and tgz.exists():
        tgz.unlink()


def sum_musdb_instrumental(
    song_dir: Path, out_wav: Path, target_sr: int | None = None, use_flac: bool = False
) -> bool:
    """Sum drums + bass + other (excluding vocals) into a single stereo WAV or FLAC."""
    import soundfile as sf  # type: ignore

    stems = [song_dir / name for name in ("drums.wav", "bass.wav", "other.wav")]
    if not all(p.exists() for p in stems):
        return False
    acc = None
    sr = 44100
    for p in stems:
        data, sr = sf.read(str(p), dtype="float32", always_2d=True)
        if acc is None:
            acc = data
        else:
            n = min(acc.shape[0], data.shape[0])
            acc = acc[:n] + data[:n]
    assert acc is not None
    peak = float(abs(acc).max())
    if peak > 0.98:
        acc = acc * (0.98 / peak)
    if target_sr is not None and sr != target_sr:
        import torch
        import torchaudio  # type: ignore

        w_t = torch.from_numpy(acc.T.copy())
        w_rs = torchaudio.functional.resample(w_t, sr, target_sr)
        rs_peak = float(w_rs.abs().max())
        if rs_peak > 0.98:
            w_rs = w_rs * (0.98 / rs_peak)
        acc = w_rs.T.numpy()
        sr = target_sr
    if use_flac:
        sf.write(str(out_wav), acc, sr, format="FLAC", subtype="PCM_16")
    else:
        sf.write(str(out_wav), acc, sr, subtype="FLOAT")
    return True


def prepare_musdb_instrumental(
    out_music: Path,
    cache_dir: Path,
    keep_archives: bool,
    target_sr: int | None = None,
    use_flac: bool = False,
    max_music: int | None = None,
) -> None:
    out_music.mkdir(parents=True, exist_ok=True)
    if list(out_music.glob("musdb_*.flac")) or list(out_music.glob("musdb_*.wav")):
        print("MUSDB18-HQ instrumental tracks already prepared, skipping.")
        return
    zpath = download_file(URLS["musdb18hq"], cache_dir / "musdb18hq.zip")
    tmp_dir = cache_dir / "_extract_musdb18hq"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    ext = ".flac" if use_flac else ".wav"
    limit = (max_music + 15) if max_music is not None else None
    print(f"Streaming MUSDB18-HQ song-by-song (drums+bass+other only, limit={limit or 'all'})...", flush=True)
    count = 0
    with zipfile.ZipFile(zpath, "r") as zf:
        by_song: dict[str, list[str]] = {}
        for m in zf.namelist():
            if m.endswith(("drums.wav", "bass.wav", "other.wav")):
                parent = str(Path(m).parent)
                by_song.setdefault(parent, []).append(m)
        song_keys = sorted(k for k, v in by_song.items() if len(v) == 3)
        for song_key in song_keys:
            if limit is not None and count >= limit:
                break
            song_p = Path(song_key)
            split = song_p.parent.name if song_p.parent.name else "track"
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in song_p.name)
            out_path = out_music / f"musdb_{split}_{safe_name}{ext}"
            tmp_song = tmp_dir / f"{split}_{safe_name}"
            try:
                tmp_song.mkdir(parents=True, exist_ok=True)
                for member in by_song[song_key]:
                    dst_m = tmp_song / Path(member).name
                    with zf.open(member) as src_f, open(dst_m, "wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f, length=1024 * 1024)
                if sum_musdb_instrumental(tmp_song, out_path, target_sr=target_sr, use_flac=use_flac):
                    count += 1
                    if count % 15 == 0:
                        print(f"  Prepared {count} MUSDB18-HQ instrumental tracks...", flush=True)
            finally:
                shutil.rmtree(tmp_song, ignore_errors=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Prepared {count} MUSDB18-HQ vocal-free instrumental tracks.", flush=True)
    if not keep_archives and zpath.exists():
        zpath.unlink()


def prepare_demand(
    env_zips: Iterable[str],
    out_amb: Path,
    cache_dir: Path,
    keep_archives: bool,
    use_flac: bool = False,
) -> None:
    out_amb.mkdir(parents=True, exist_ok=True)
    for zname in env_zips:
        env_tag = zname.replace(".zip", "")
        if (out_amb / f"demand_{env_tag}_ch01.wav").exists() or (out_amb / f"demand_{env_tag}_ch01.flac").exists():
            continue
        url = f"https://zenodo.org/api/records/1227121/files/{zname}/content"
        tmp_dir = cache_dir / f"_extract_{env_tag}"
        try:
            zpath = download_file(url, cache_dir / zname)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            with zipfile.ZipFile(zpath, "r") as zf:
                members = [m for m in zf.namelist() if m.endswith("ch01.wav") or m.endswith("ch09.wav")]
                for m in members:
                    zf.extract(m, tmp_dir)
            if use_flac:
                import soundfile as sf  # type: ignore

                for ch_file in sorted(tmp_dir.rglob("*.wav")):
                    data, sr = sf.read(str(ch_file), dtype="float32", always_2d=True)
                    pk = float(abs(data).max())
                    if pk > 0.99:
                        data = data * (0.99 / pk)
                    sf.write(str(out_amb / f"demand_{env_tag}_{ch_file.stem}.flac"), data, sr, format="FLAC", subtype="PCM_16")
            else:
                for ch_file in sorted(tmp_dir.rglob("*.wav")):
                    shutil.move(str(ch_file), str(out_amb / f"demand_{env_tag}_{ch_file.name}"))
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if not keep_archives and zpath.exists():
                zpath.unlink()
        except Exception as exc:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"Warning: skipping DEMAND archive {zname} due to error: {exc}", flush=True)


def prepare_rirs(out_rir: Path, cache_dir: Path, keep_archives: bool, max_rirs: int = 2000) -> None:
    out_rir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_rir.glob("*.wav"))
    if existing:
        if len(existing) > max_rirs:
            step = max(1, len(existing) // max_rirs)
            keep_set = set(existing[::step][:max_rirs])
            removed = 0
            for p in existing:
                if p not in keep_set:
                    p.unlink()
                    removed += 1
            print(f"Trimmed RIRs from {len(existing)} to {len(keep_set)} (removed {removed}).", flush=True)
        else:
            print(f"RIR dir already populated ({len(existing)} files), skipping.", flush=True)
        return
    zpath = download_file(URLS["rirs_noises"], cache_dir / "rirs_noises.zip")
    tmp_dir = cache_dir / "_extract_rirs"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    print("Extracting simulated RIRs from SLR28...", flush=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        members = sorted(m for m in zf.namelist() if "simulated_rirs/" in m and m.endswith(".wav"))
        if len(members) > max_rirs:
            step = max(1, len(members) // max_rirs)
            members = members[::step][:max_rirs]
        for m in members:
            zf.extract(m, tmp_dir)
    count = 0
    for idx, wav in enumerate(sorted(tmp_dir.rglob("*.wav"))):
        shutil.move(str(wav), str(out_rir / f"sim_rir_{idx:05d}.wav"))
        count += 1
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Prepared {count} simulated RIR files.", flush=True)
    if not keep_archives and zpath.exists():
        zpath.unlink()


def ensure_all_48k(
    out_dir: Path,
    target_sr: int = 48000,
    subdirs: tuple[str, ...] = ("speech", "music", "amb", "rir"),
) -> None:
    """Resample any non-48 kHz files in selected stems to 48000 Hz in-place."""
    import soundfile as sf  # type: ignore
    import torch
    import torchaudio  # type: ignore

    converted = 0
    for sub in subdirs:
        d = out_dir / sub
        if not d.exists():
            continue
        files = [p for p in sorted(d.rglob("*")) if p.is_file() and p.suffix.lower() in {".wav", ".flac"}]
        sub_conv = 0
        for p in files:
            info = sf.info(str(p))
            if info.samplerate == target_sr:
                continue
            wav, src_sr = sf.read(str(p), dtype="float32", always_2d=True)
            w_t = torch.from_numpy(wav.T.copy())
            w_rs = torchaudio.functional.resample(w_t, src_sr, target_sr)
            peak = float(w_rs.abs().max())
            if peak > 0.99:
                w_rs = w_rs * (0.99 / peak)
            out_arr = w_rs.T.numpy()
            tmp = p.with_name(f"{p.stem}._48k{p.suffix}")
            subtype = "PCM_16" if p.suffix.lower() in {".wav", ".flac"} else None
            sf.write(str(tmp), out_arr, target_sr, subtype=subtype)
            tmp.replace(p)
            converted += 1
            sub_conv += 1
        print(f"ensure_all_48k [{sub}]: resampled {sub_conv}/{len(files)} files to {target_sr} Hz.", flush=True)
    print(f"ensure_all_48k complete: {converted} total files converted to {target_sr} Hz.", flush=True)


def verify_dataset(out_dir: Path) -> None:
    """Quick local audio integrity & count check (no GPU/Colab units used)."""
    import soundfile as sf  # type: ignore

    for sub in ("speech", "music", "amb", "rir"):
        d = out_dir / sub
        if not d.exists():
            continue
        files = [p for p in d.rglob("*") if p.suffix.lower() in {".wav", ".flac"}]
        bad = 0
        non_48k = 0
        total_bytes = sum(p.stat().st_size for p in files)
        # Check every file header via soundfile.info (fast, reads header only)
        for p in files:
            try:
                info = sf.info(str(p))
                if info.frames <= 0 or info.samplerate <= 0:
                    bad += 1
                elif info.samplerate != 48000:
                    non_48k += 1
            except Exception:
                bad += 1
        extra = ""
        if sub == "speech":
            spks = [x for x in d.iterdir() if x.is_dir()]
            lang_counts = {
                lang: sum(1 for x in spks if x.name.startswith(f"{lang}_"))
                for lang in ("en", "ta", "te", "zh", "hi")
            }
            lang_str = ", ".join(f"{k}={v}" for k, v in lang_counts.items())
            extra = f" | speakers={len(spks)} ({lang_str})"
        print(
            f"Verified {sub:6s}: {len(files):6d} files | {total_bytes/1e9:.2f} GB{extra} | "
            f"non_48k={non_48k} | corrupt={bad}",
            flush=True,
        )


def ml_audit_background(out_dir: Path, max_music: int | None = None) -> None:
    """Strictly audit music (zero vocals/speech) and ambience (zero voice/music) using Silero VAD + AudioSet AST."""
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore
    import torch
    import torchaudio  # type: ignore
    try:
        from transformers import ASTForAudioClassification, AutoFeatureExtractor  # type: ignore
    except ImportError:
        print("Warning: transformers not installed; skipping ML background purity audit.", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running ML background purity audit on {out_dir} (device={device})...", flush=True)
    vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    vad_model = vad_model.to("cpu").eval()

    ast_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
    ast_fe = AutoFeatureExtractor.from_pretrained(ast_id)
    ast_model = ASTForAudioClassification.from_pretrained(ast_id, use_safetensors=True).to(device).eval()
    l2i = ast_model.config.label2id

    speech_names = [
        "Speech", "Male speech, man speaking", "Female speech, woman speaking",
        "Child speech, kid speaking", "Conversation", "Narration, monologue",
        "Speech synthesizer", "Chatter", "Hubbub, speech noise, speech babble",
    ]
    sing_names = [
        "Singing", "Choir", "Male singing", "Female singing",
        "Child singing", "Synthetic singing", "Vocal music",
    ]
    music_names = [
        k for k in l2i
        if ("music" in k.lower() and k not in ("Singing bowl", "Vocal music")) or k in ("Music", "Musical instrument")
    ]
    speech_ids = [l2i[k] for k in speech_names if k in l2i]
    sing_ids = [l2i[k] for k in sing_names if k in l2i]
    music_ids = [l2i[k] for k in music_names if k in l2i]

    def _scan_file(path: Path) -> tuple[float, float, float, float, float]:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = wav.mean(axis=1)
        if sr != 16000:
            mono = torchaudio.functional.resample(torch.from_numpy(mono).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        vad_model.reset_states()
        n_frames = len(mono) // 512
        longest = 0
        cur = 0
        if n_frames > 0:
            frames = torch.from_numpy(mono[: n_frames * 512].reshape(n_frames, 512))
            for i in range(n_frames):
                p = float(vad_model(frames[i : i + 1], 16000).item())
                if p >= 0.35:
                    cur += 1
                    if cur > longest:
                        longest = cur
                else:
                    cur = 0
        vad_longest_ms = float(longest * 32.0)

        wlen = 160000
        windows = []
        if len(mono) <= wlen:
            windows.append(mono)
        else:
            st = 0
            for st in range(0, len(mono) - wlen + 1, wlen):
                windows.append(mono[st : st + wlen])
            if len(mono) - (st + wlen) > 32000:
                windows.append(mono[-wlen:])

        sp_list, sg_list, mu_list = [], [], []
        with torch.no_grad():
            for i in range(0, len(windows), 8):
                batch = windows[i : i + 8]
                inp = ast_fe(batch, sampling_rate=16000, return_tensors="pt")
                probs = torch.sigmoid(ast_model(inp["input_values"].to(device)).logits).cpu()
                sp_list.extend(probs[:, speech_ids].max(dim=1).values.tolist())
                sg_list.extend(probs[:, sing_ids].max(dim=1).values.tolist())
                mu_list.extend(probs[:, music_ids].max(dim=1).values.tolist())
        return (
            vad_longest_ms,
            float(max(sp_list)),
            float(np.mean(sp_list)),
            float(max(sg_list)),
            float(max(mu_list)),
        )

    music_dir = out_dir / "music"
    if music_dir.exists():
        music_files = [p for p in sorted(music_dir.rglob("*")) if p.is_file() and p.suffix.lower() in {".wav", ".flac"}]
        clean_music: list[tuple[float, Path]] = []
        removed_music = 0
        for p in music_files:
            vad_ms, sp_max, _sp_mean, sg_max, _mu_max = _scan_file(p)
            if sg_max >= 0.30 or sp_max >= 0.45 or (vad_ms >= 350.0 and sp_max >= 0.20):
                print(f"  [REJECT music] {p.name}: sing={sg_max:.2f} speech={sp_max:.2f} vad={vad_ms:.0f}ms", flush=True)
                p.unlink()
                removed_music += 1
            else:
                clean_music.append((sg_max + sp_max, p))
        if max_music is not None and len(clean_music) > max_music:
            clean_music.sort(key=lambda x: x[0])
            for _score, extra_p in clean_music[max_music:]:
                extra_p.unlink()
            clean_music = clean_music[:max_music]
        print(f"ML audit [music]: kept {len(clean_music)} / {len(music_files)} tracks (rejected {removed_music} for vocal risk).", flush=True)

    amb_dir = out_dir / "amb"
    if amb_dir.exists():
        amb_files = [p for p in sorted(amb_dir.rglob("*")) if p.is_file() and p.suffix.lower() in {".wav", ".flac"}]
        kept_amb = 0
        removed_amb = 0
        for p in amb_files:
            vad_ms, sp_max, sp_mean, sg_max, mu_max = _scan_file(p)
            if vad_ms >= 200.0 or sp_mean >= 0.15 or sp_max >= 0.55 or mu_max >= 0.25 or sg_max >= 0.15:
                print(f"  [REJECT amb] {p.name}: vad={vad_ms:.0f}ms sp_max={sp_max:.2f} sp_mean={sp_mean:.2f} mu_max={mu_max:.2f}", flush=True)
                p.unlink()
                removed_amb += 1
            else:
                kept_amb += 1
        print(f"ML audit [amb]: kept {kept_amb} / {len(amb_files)} files (rejected {removed_amb} for voice/music risk).", flush=True)

    del vad_model, ast_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pack_tar_for_gdrive(
    out_dir: Path,
    tar_dir: Path,
    items: list[tuple[str, str]] | None = None,
) -> None:
    """Atomically pack stem folders into uncompressed .tar archives (safe on Drive space limits)."""
    tar_dir.mkdir(parents=True, exist_ok=True)
    if items is None:
        items = [(s, f"{s}.tar") for s in ("speech", "music", "amb", "rir")]
    for sub, tar_name in items:
        src = out_dir / sub
        if not src.exists() or not any(src.iterdir()):
            continue
        tar_path = tar_dir / tar_name
        tmp_path = tar_dir / f"{tar_name}.tmp"
        print(f"Packing {src} -> {tar_path}...", flush=True)
        try:
            with tarfile.open(tmp_path, "w") as tf:
                tf.add(src, arcname=sub)
            os.replace(tmp_path, tar_path)
            print(f"  Created {tar_path.name} ({tar_path.stat().st_size/1e9:.2f} GB)", flush=True)
        except OSError as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"  Warning: could not pack {tar_name} to {tar_dir} ({exc}); continuing from {src}.", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Download & organize public speech/music/ambience/RIR datasets.")
    p.add_argument("--preset", choices=["quick", "ablation_16k", "full_48k", "upgrade_bg_48k"], default="ablation_16k")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--keep-archives", action="store_true")
    p.add_argument("--no-purify", action="store_true", help="Skip strict speech noise-floor/SNR purification")
    p.add_argument("--max-music", type=int, default=60, help="Max clean instrumental tracks to keep for upgrade_bg_48k")
    p.add_argument("--audit", action="store_true", help="Run Silero VAD + AudioSet AST purity audit on music & amb")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--pack-tar", type=Path, default=None, help="Directory to write <stem>.tar files for GDrive upload")
    args = p.parse_args()

    out = args.out
    cache = out / "_cache"
    out_speech = out / "speech"
    out_music = out / "music"
    out_amb = out / "amb"
    out_rir = out / "rir"

    if args.preset == "upgrade_bg_48k":
        prepare_musdb_instrumental(
            out_music, cache, args.keep_archives,
            target_sr=48000, use_flac=True, max_music=args.max_music,
        )
        if len(list(out_music.glob("musdb_*.flac"))) >= min(30, args.max_music):
            removed_m = 0
            for legacy_m in list(out_music.glob("babyslakh_*.wav")):
                legacy_m.unlink()
                removed_m += 1
            if removed_m:
                print(f"Removed {removed_m} legacy 16 kHz upsampled BabySlakh tracks from {out_music}.", flush=True)
        prepare_demand(DEMAND_CLEAN_48K, out_amb, cache, args.keep_archives, use_flac=True)
        if len(list(out_amb.glob("demand_*_48k_*.flac"))) >= 6:
            removed_a = 0
            for legacy_a in list(out_amb.glob("demand_*_16k_*.wav")):
                legacy_a.unlink()
                removed_a += 1
            if removed_a:
                print(f"Removed {removed_a} legacy 16 kHz upsampled DEMAND tracks from {out_amb}.", flush=True)
        ensure_all_48k(out, target_sr=48000, subdirs=("music", "amb"))
        ml_audit_background(out, max_music=args.max_music)
    elif args.preset == "full_48k":
        prepare_multilingual_48k(out_speech, cache, args.keep_archives)
        seed_speech = Path(r"E:\AudioSep_Dataset\data\speech")
        if seed_speech.exists() and not any(out_speech.glob("en_ls_*")):
            print(f"Seeding 20 additional single-speaker folders from {seed_speech}...", flush=True)
            for spk_d in sorted(d for d in seed_speech.iterdir() if d.is_dir())[:20]:
                dst_d = out_speech / f"en_ls_{spk_d.name}"
                dst_d.mkdir(parents=True, exist_ok=True)
                for f in sorted(p for p in spk_d.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac"})[:25]:
                    shutil.copy2(f, dst_d / f.name)
        if not args.no_purify:
            purify_speech(
                out_speech,
                min_snr_db=28.0,
                max_floor_db=-42.0,
                min_files_per_spk=5,
                max_files_per_spk=25,
                max_spks_per_lang=42,
            )
        seed_music = Path(r"E:\AudioSep_Dataset\data\music")
        out_music.mkdir(parents=True, exist_ok=True)
        if seed_music.exists() and not any(out_music.iterdir()):
            print(f"Seeding clean instrumental music from {seed_music}...")
            for f in sorted(seed_music.glob("*")):
                if f.is_file():
                    shutil.copy2(f, out_music / f.name)
        elif not any(out_music.iterdir()):
            prepare_babyslakh(out_music, cache, args.keep_archives)

        seed_amb = Path(r"E:\AudioSep_Dataset\data\amb")
        out_amb.mkdir(parents=True, exist_ok=True)
        if seed_amb.exists() and not any(out_amb.iterdir()):
            amb_files = sorted(f for f in seed_amb.glob("*.wav") if f.is_file())[:10]
            print(f"Seeding {len(amb_files)} DEMAND ambience tracks from {seed_amb}...", flush=True)
            for f in amb_files:
                shutil.copy2(f, out_amb / f.name)
        elif not any(out_amb.iterdir()):
            prepare_demand(DEMAND_QUICK_48K, out_amb, cache, args.keep_archives)

        seed_rir = Path(r"E:\AudioSep_Dataset\data\rir")
        out_rir.mkdir(parents=True, exist_ok=True)
        max_rirs = 400
        if seed_rir.exists() and not any(out_rir.iterdir()):
            all_rirs = sorted(f for f in seed_rir.glob("*") if f.is_file())
            step = max(1, len(all_rirs) // max_rirs)
            picked = all_rirs[::step][:max_rirs]
            print(f"Seeding {len(picked)} RIRs (from {len(all_rirs)}) from {seed_rir}...")
            for f in picked:
                shutil.copy2(f, out_rir / f.name)
        else:
            prepare_rirs(out_rir, cache, args.keep_archives, max_rirs=max_rirs)

        ensure_all_48k(out, target_sr=48000)
    else:
        url_key, subset_name = ("dev_clean", "dev-clean") if args.preset == "quick" else ("train_clean_100", "train-clean-100")
        prepare_librispeech(URLS[url_key], subset_name, out_speech, cache, args.keep_archives)
        if not args.no_purify:
            purify_speech(out_speech)

        if args.preset == "quick":
            prepare_babyslakh(out_music, cache, args.keep_archives)
            prepare_demand(DEMAND_QUICK_16K, out_amb, cache, args.keep_archives)
        else:  # ablation_16k
            prepare_babyslakh(out_music, cache, args.keep_archives)
            prepare_demand(DEMAND_ALL_16K, out_amb, cache, args.keep_archives)
            prepare_rirs(out_rir, cache, args.keep_archives)

    if cache.exists() and not any(cache.iterdir()):
        cache.rmdir()
    if args.audit and args.preset != "upgrade_bg_48k":
        ml_audit_background(out)
    if args.verify:
        verify_dataset(out)
    if args.pack_tar is not None:
        pack_items = (
            [("music", "music_hq48.tar"), ("amb", "amb_48k.tar")]
            if args.preset == "upgrade_bg_48k"
            else None
        )
        pack_tar_for_gdrive(out, args.pack_tar, items=pack_items)
    print(f"\nDone! Dataset ready under: {out.resolve()}")


if __name__ == "__main__":
    main()
