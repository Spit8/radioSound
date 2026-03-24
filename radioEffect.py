#!/usr/bin/env python3
"""
Pipeline audio "Effet Radio / Diffusion TV"
==========================================
Étapes :
  1. Bandpass filter (300 Hz – 3 kHz)
  2. Ajout de bruit blanc (statique)
  3. Compression dynamique forte
  4. EQ radio (boost médiums, coupe graves/aigus)

Dépendances (100% pip, aucun outil système requis) :
    pip install numpy scipy soundfile imageio-ffmpeg

Usage:
python radioEffect.py input.mp3 output.mp3

Conseils de réglages:
    --noise     : 0.001 (subtil) à 0.01 (prononcé)
    --threshold : 0.1 (compression forte) à 0.4 (compression légère)
    --ratio     : 4 (compression légère) à 10 (compression très forte)
    
Pour un effet "radio":
--noise 0.006 --threshold 0.15 --ratio 10
Pour un effet "Intercom/Talkie-Walkie":
--noise 0.01 --threshold 0.1 --ratio 15
Pour un effet "TV ancienne":
--noise 0.002 --ratio 6
"""

import argparse
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pathlib import Path


# ─────────────────────────────────────────────
# Utilitaire : binaire ffmpeg embarqué dans le venv
# ─────────────────────────────────────────────

def _ffmpeg_exe() -> str:
    """
    Retourne le chemin absolu vers le binaire ffmpeg fourni par
    imageio-ffmpeg (embarqué dans le venv, aucune installation système requise).
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise ImportError(
            "\n❌  imageio-ffmpeg introuvable.\n"
            "    Lance : pip install imageio-ffmpeg\n"
        )


# ─────────────────────────────────────────────
# 1. Chargement / sauvegarde
# ─────────────────────────────────────────────

def load_audio(path: str):
    """
    Charge n'importe quel format audio supporté par ffmpeg
    (mp3, wav, ogg, flac, aac…) → numpy float32 mono, sample rate.
    Utilise le binaire ffmpeg embarqué par imageio-ffmpeg.
    """
    ffmpeg = _ffmpeg_exe()
    path   = Path(path)

    # 1) Tente de récupérer le sample rate via ffprobe (même dossier que ffmpeg)
    ffprobe = Path(ffmpeg).parent / "ffprobe"
    sr = 44100  # valeur par défaut si ffprobe absent
    if ffprobe.exists():
        try:
            probe = subprocess.run(
                [
                    str(ffprobe), "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=sample_rate",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True, text=True, check=True,
            )
            sr = int(probe.stdout.strip())
        except Exception:
            pass  # ffprobe indisponible → on garde 44100

    # 2) Décode en PCM s16le mono via pipe
    proc = subprocess.run(
        [
            ffmpeg, "-v", "error",
            "-i", str(path),
            "-ac", "1",        # mono
            "-ar", str(sr),    # sample rate
            "-f", "s16le",     # PCM 16-bit little-endian
            "-",               # stdout
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    samples /= 32768.0         # normalise en [-1.0, 1.0]
    return samples, sr


def save_audio(path: str, samples: np.ndarray, sr: int):
    """Sauvegarde en .wav ou .mp3 via le binaire ffmpeg embarqué."""
    ffmpeg  = _ffmpeg_exe()
    path    = Path(path)
    samples = np.clip(samples, -1.0, 1.0)
    pcm     = (samples * 32767).astype(np.int16)

    cmd = [
        ffmpeg, "-y",          # écrase sans demander
        "-v", "error",
        "-f", "s16le",
        "-ar", str(sr),
        "-ac", "1",
        "-i", "pipe:0",        # lecture depuis stdin
    ]

    if path.suffix.lower() == ".mp3":
        cmd += ["-b:a", "128k"]

    cmd.append(str(path))

    subprocess.run(cmd, input=pcm.tobytes(), check=True)
    print(f"  ✔ Fichier sauvegardé : {path}")


# ─────────────────────────────────────────────
# 2. Bandpass filter 300 Hz – 3 kHz
# ─────────────────────────────────────────────

def bandpass_filter(samples: np.ndarray, sr: int,
                    low_hz: float = 300.0,
                    high_hz: float = 3000.0,
                    order: int = 6) -> np.ndarray:
    """Filtre passe-bande Butterworth (SOS pour la stabilité numérique)."""
    nyq = sr / 2.0
    sos = butter(order, [low_hz / nyq, high_hz / nyq],
                 btype="band", output="sos")
    return sosfilt(sos, samples)


# ─────────────────────────────────────────────
# 3. Ajout de bruit blanc (statique)
# ─────────────────────────────────────────────

def add_white_noise(samples: np.ndarray,
                    noise_level: float = 0.004) -> np.ndarray:
    """Ajoute du bruit blanc gaussien au signal."""
    noise = np.random.normal(0.0, noise_level, size=samples.shape)
    return samples + noise


# ─────────────────────────────────────────────
# 4. Compression dynamique forte
# ─────────────────────────────────────────────

def compress(samples: np.ndarray,
             threshold: float = 0.2,
             ratio: float = 8.0,
             attack_ms: float = 5.0,
             release_ms: float = 50.0,
             sr: int = 44100,
             makeup_gain: float = 2.5) -> np.ndarray:
    """
    Compresseur dynamique sample-par-sample.
      threshold   : niveau (0-1) au-delà duquel la compression s'active
      ratio       : taux de compression (ex: 8 → 8:1)
      attack_ms   : temps d'attaque en millisecondes
      release_ms  : temps de relâchement en millisecondes
      makeup_gain : gain de compensation appliqué après compression
    """
    attack_coeff  = np.exp(-1.0 / (sr * attack_ms  / 1000.0))
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000.0))

    envelope = 0.0
    out = np.zeros_like(samples)

    for i, x in enumerate(samples):
        level = abs(x)
        if level > envelope:
            envelope = attack_coeff  * envelope + (1 - attack_coeff)  * level
        else:
            envelope = release_coeff * envelope + (1 - release_coeff) * level

        if envelope > threshold:
            gain_reduction = threshold + (envelope - threshold) / ratio
            gain = gain_reduction / (envelope + 1e-9)
        else:
            gain = 1.0

        out[i] = x * gain

    return out * makeup_gain


# ─────────────────────────────────────────────
# 5. EQ Radio
# ─────────────────────────────────────────────

def eq_radio(samples: np.ndarray, sr: int) -> np.ndarray:
    """
    Égalisation "radio" :
      - Coupe les sub-graves  (<150 Hz)     passe-haut 2nd ordre
      - Boost léger des médiums (1–2.5 kHz) peak EQ additif
      - Coupe douce des hauts aigus (>3.5 kHz) passe-bas
    """
    nyq = sr / 2.0

    # Passe-haut : supprime les sub-graves
    sos_hp = butter(2, 150.0 / nyq, btype="high", output="sos")
    samples = sosfilt(sos_hp, samples)

    # Passe-bas : adoucit les très hauts aigus
    sos_lp = butter(4, 3500.0 / nyq, btype="low", output="sos")
    samples = sosfilt(sos_lp, samples)

    # Boost médiums : filtre passe-bande additionné au signal (+~3 dB)
    sos_mid = butter(2, [1000.0 / nyq, 2500.0 / nyq], btype="band", output="sos")
    mid_boost = sosfilt(sos_mid, samples)
    samples = samples + 0.4 * mid_boost

    return samples


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

def radio_pipeline(input_path: str,
                   output_path: str,
                   noise_level: float = 0.004,
                   comp_threshold: float = 0.2,
                   comp_ratio: float = 8.0):
    """
    Applique le pipeline complet effet radio/TV.

    Paramètres ajustables :
      noise_level      : intensité du bruit blanc   (0.001 = subtil, 0.01 = prononcé)
      comp_threshold   : seuil de compression       (0.1 – 0.4)
      comp_ratio       : ratio de compression       (4 = léger, 10 = très fort)
    """
    print(f"\n📻  Pipeline Radio / TV")
    print(f"  Entrée  : {input_path}")
    print(f"  Sortie  : {output_path}")
    print()

    # ── Chargement ──────────────────────────────────────────────────────────
    print("  [1/5] Chargement du fichier audio…")
    samples, sr = load_audio(input_path)
    print(f"        {len(samples)/sr:.1f}s  |  {sr} Hz  |  {len(samples)} samples")

    # ── Bandpass 300 Hz – 3 kHz ─────────────────────────────────────────────
    print("  [2/5] Filtre passe-bande 300 Hz – 3 kHz…")
    samples = bandpass_filter(samples, sr, low_hz=300.0, high_hz=3000.0)

    # ── Bruit blanc ──────────────────────────────────────────────────────────
    print(f"  [3/5] Ajout de bruit blanc (niveau={noise_level})…")
    samples = add_white_noise(samples, noise_level=noise_level)

    # ── Compression ─────────────────────────────────────────────────────────
    print(f"  [4/5] Compression dynamique (threshold={comp_threshold}, ratio={comp_ratio}:1)…")
    samples = compress(samples,
                       threshold=comp_threshold,
                       ratio=comp_ratio,
                       sr=sr)

    # ── EQ Radio ────────────────────────────────────────────────────────────
    print("  [5/5] EQ Radio (coupe graves/aigus, boost médiums)…")
    samples = eq_radio(samples, sr)

    # ── Sauvegarde ──────────────────────────────────────────────────────────
    print("\n  Sauvegarde…")
    save_audio(output_path, samples, sr)
    print("\n✅  Traitement terminé !\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Applique un effet Radio / Diffusion TV à un fichier audio."
    )
    parser.add_argument("input",  help="Fichier source (.mp3 ou .wav)")
    parser.add_argument("output", help="Fichier de sortie (.mp3 ou .wav)")
    parser.add_argument(
        "--noise", type=float, default=0.004,
        help="Niveau du bruit blanc (défaut: 0.004)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="Seuil de compression 0-1 (défaut: 0.2)"
    )
    parser.add_argument(
        "--ratio", type=float, default=8.0,
        help="Ratio de compression (défaut: 8.0)"
    )

    args = parser.parse_args()

    radio_pipeline(
        input_path=args.input,
        output_path=args.output,
        noise_level=args.noise,
        comp_threshold=args.threshold,
        comp_ratio=args.ratio,
    )
