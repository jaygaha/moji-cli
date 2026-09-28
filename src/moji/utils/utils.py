# src/utils/utils.py

from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional
import logging
import shutil

# Module-level logger. __name__ becomes "src.utils.utils",
# so it inherits configuration from the root logger.
LOG = logging.getLogger(__name__)

def run_cmd(cmd: List[str], timeout: Optional[int] = None) -> str:
    """Run an external command and return its stdout.

    Raises RuntimeError (with truncated stderr) on non-zero exit.
    Logs the full command at DEBUG level.
    """
    LOG.debug("CMD: %s", " ".join(map(str, cmd)))

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )

    if proc.returncode != 0:
        LOG.error("Command failed with code %s:", proc.returncode)
        LOG.error("STDOUT: %s", proc.stdout)
        LOG.error("STDERR: %s", proc.stderr)

        raise RuntimeError(
            f"Command failed with code {proc.returncode}: {' '.join(map(str, cmd))}\n"
            f"{proc.stderr[-2000:]}"
        )

    return proc.stdout.strip()
    
def require_binary(name: str) -> str:
    """Return the absolute path of an executable found in PATH.

    Raises RuntimeError if the binary is missing.
    """
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Binary '{name}' not found in PATH")

    return path
    
def media_duration(path: Path) -> float:
    """Return the duration (seconds) of a media file via ffprobe."""
    out = run_cmd(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )

    return float(out)
    
def extract_audio(
    input_path: Path, 
    workdir: Path, 
    audio_name: Optional[str] = None, 
    denoise: bool = False
) -> Path:
    """Extract the first audio stream as 16 kHz mono PCM WAV.

    Optionally applies a conservative speech-oriented denoise filter.
    Raises RuntimeError if the output file is missing or has zero duration.
    """
    wav = workdir / f"{audio_name or input_path.stem}.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(input_path),
        "-vn",
        "-map", "0:a:0",
        "-ac", "1",
        "-ar", "16000",
    ]
    if denoise:
        # highpass + lowpass + mild FFT denoise; tune or disable if artefacts appear
        cmd.extend(["-af","highpass=f=80,lowpass=f=8000,afftdn=nf=-25"])

    cmd.extend(["-c:a", "pcm_s16le", str(wav)])
    run_cmd(cmd)
    
    if not wav.exists():
        raise RuntimeError(f"Audio extraction failed: {wav} does not exist.")

    dur = media_duration(wav)

    if dur <= 0:
        raise RuntimeError("Extracted audio has zero duration.")


    return wav

def split_audio(
    wav: Path,
    workdir: Path,
    duration: float,
    chunk_sec: float,
    overlap_sec: float,
) -> List[Dict[str, Any]]:
    """Split a WAV into overlapping chunks for sequential processing.

    Each returned dict contains:
      - index          : sequential chunk number
      - path           : Path to the chunk WAV
      - nominal_start  : logical start time in the original audio
      - nominal_end    : logical end time in the original audio

    The first chunk starts at 0; subsequent chunks begin `overlap_sec`
    earlier so the ASR model has context across boundaries.
    """
    if overlap_sec >= chunk_sec:
        raise ValueError("--overlap-seconds must be smaller than --chunk-seconds")

    chunks: List[Dict[str, Any]] = []
    nominal_start = 0.0
    index = 0

    while nominal_start < duration:
        nominal_end = min(duration, nominal_start + chunk_sec)

        # Pull a little earlier for context on every chunk after the first
        extract_start = max(0.0, nominal_start - overlap_sec) if index > 0 else 0.0
        extract_end = nominal_end

        if extract_end <= extract_start:
            nominal_start += chunk_sec
            continue

        out = workdir / f"chunk_{index:03d}.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-ss", f"{extract_start:.3f}",
            "-t", f"{extract_end - extract_start:.3f}",
            "-i", str(wav),
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(out),
        ]

        run_cmd(cmd)

        if not out.exists():
            raise RuntimeError(f"ffmpeg failed to create chunk file: {out}")

        chunks.append({
            "index": index,
            "path": out,
            "nominal_start": nominal_start,
            "nominal_end": nominal_end,
            "extract_start": extract_start,
        })

        if nominal_end >= duration:
            break
        
        nominal_start = nominal_end
        index += 1

    return chunks
    