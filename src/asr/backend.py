# src/asr/backend.py

import json
import argparse
import shutil

from pathlib import Path
from typing import List, Any, Dict, Optional
from dataclasses import asdict

from src.data.models import Segment
from src.utils.utils import run_cmd

def determine_asr_backend(args: argparse.Namespace) -> str:
    """Select the ASR backend.

    Preference order when --asr-backend=auto:
      1. Python package mlx_whisper
      2. mlx_whisper CLI binary on PATH
    Raises RuntimeError if neither is available.
    """
    if args.asr_backend != "auto":
        return args.asr_backend
    
    try:
        import mlx_whisper  # noqa: F401
        return "python"

    except Exception:
        pass

    if shutil.which(args.mlx_cli):
        return "cli"

    raise RuntimeError(
        "No usable mlx-whisper backend found.\n"
        "Install Python mlx-whisper: pip install mlx-whisper\n"
        "or ensure mlx_whisper CLI is on PATH, or pass --asr-backend cli."
    )

def mlx_python_transcribe(audio_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Transcribe via the Python mlx_whisper API.

    Tries a richer set of options first, then falls back to a minimal
    argument set for broader API compatibility across mlx-whisper versions.
    """
    import mlx_whisper

    common = {
        "language": args.language,
        "task": "transcribe",
        "word_timestamps": False,
        "verbose": False,
    }

    full = {
        **common,
        "condition_on_previous_text": False,
        "temperature": 0.0,
    }

    if args.initial_prompt:
        full["initial_prompt"] = args.initial_prompt

    last_error: Optional[Exception] = None

    # Try richer options first, then minimal options for API compatibility.
    for kwargs in (full, common):
        try:
            if args.asr_model:
                try:
                    return mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=args.asr_model, **kwargs)
                except TypeError:
                    # Some releases accept the model only as a keyword.
                    return mlx_whisper.transcribe(
                        str(audio_path),
                        path_or_hf=args.asr_model,
                        **kwargs,
                    )
            return mlx_whisper.transcribe(str(audio_path), **kwargs)
        except TypeError as exc:
            last_error = exc
            continue

    raise RuntimeError(f"mlx_whisper.transcribe failed: {last_error}")

def mlx_cli_transcribe(audio_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Transcribe via the mlx_whisper CLI and return the parsed JSON result.

    Looks for a JSON file written to a sibling .out directory; falls back to
    parsing JSON from stdout when the CLI emits it directly.
    """
    outdir = Path(str(audio_path) + ".out")
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.mlx_cli,
        str(audio_path),
        "--language", args.language,
        "--task", "transcribe",
        "--output-format", "json",
        "--output-dir", str(outdir),
    ]

    if args.asr_model:
        cmd += ["--model", args.asr_model]

    if args.initial_prompt:
        # Not all CLI builds support this flag; caller may need to retry without it.
        cmd += ["--initial-prompt", args.initial_prompt]

    stdout = run_cmd(cmd)

    json_files = sorted(outdir.glob("*.json"))
    if json_files:
        return json.loads(json_files[0].read_text(encoding="utf-8"))

    # Some CLI variants print JSON to stdout instead of writing a file.
    stripped = stdout.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(stdout)

    raise RuntimeError(f"mlx CLI did not produce expected JSON output for {audio_path}")

def transcribe_chunk(
    audio_path: Path,
    backend: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Dispatch a single audio chunk to the chosen backend."""
    if backend == "python":
        return mlx_python_transcribe(audio_path, args)
    if backend == "cli":
        return mlx_cli_transcribe(audio_path, args)
    raise ValueError(f"Unsupported ASR backend: {backend}")


def normalize_asr_segments(
    result: Dict[str, Any],
    chunk: Dict[str, Any],
    next_id: int,
) -> tuple[List[Segment], int]:
    """Convert raw ASR segments into model Segment objects with absolute times.

    Required keys in `chunk`:
      - extract_start  : actual start time of the audio file fed to ASR
      - nominal_start  : logical start of this chunk in the original media
      - nominal_end    : logical end of this chunk in the original media

    Segments whose midpoint falls outside the nominal interval (with a small
    epsilon) are discarded so that overlapped regions are not double-counted.
    """
    out: List[Segment] = []

    offset = float(chunk["extract_start"])
    nominal_start = float(chunk["nominal_start"])
    nominal_end = float(chunk["nominal_end"])

    raw_segments = result.get("segments", []) if isinstance(result, dict) else []

    for raw in raw_segments:
        text = (raw.get("text") or "").strip()
        if not text:
            continue

        raw_start = float(raw.get("start", 0.0))
        raw_end = float(raw.get("end", raw_start))

        start = raw_start + offset
        end = raw_end + offset

        if end < start:
            end = start

        mid = (start + end) / 2.0
        eps = 0.02

        # Keep segments whose midpoint belongs to this chunk's nominal interval.
        if mid < nominal_start - eps or mid >= nominal_end + eps:
            continue

        no_speech = float(raw.get("no_speech_prob", 0.0) or 0.0)
        confidence = max(0.0, min(1.0, 1.0 - no_speech))

        out.append(
            Segment(
                id=next_id,
                start=max(0.0, start),
                end=max(start, end),
                ja=text,
                confidence=round(confidence, 4),
                avg_logprob=raw.get("avg_logprob"),
                no_speech_prob=raw.get("no_speech_prob"),
                compression_ratio=raw.get("compression_ratio"),
            )
        )
        next_id += 1

    return out, next_id


def sort_and_clean_segments(segments: List[Segment]) -> List[Segment]:
    """Sort by time and merge near-identical consecutive segments.

    Two segments are merged when they are almost contiguous (< 50 ms gap)
    and contain the exact same text. IDs are re-assigned sequentially afterwards.
    """
    segments.sort(key=lambda s: (s.start, s.end))

    cleaned: List[Segment] = []
    for seg in segments:
        if (
            cleaned
            and abs(seg.start - cleaned[-1].end) < 0.05
            and seg.ja == cleaned[-1].ja
        ):
            cleaned[-1].end = max(cleaned[-1].end, seg.end)
        else:
            cleaned.append(seg)

    for i, seg in enumerate(cleaned):
        seg.id = i

    return cleaned


def save_segments(path: Path, segments: List[Segment]) -> None:
    """Serialize segments to a UTF-8 JSON file (pretty-printed)."""
    path.write_text(
        json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_segments(path: Path) -> List[Segment]:
    """Load segments previously written by save_segments."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Segment.from_dict(d) for d in data]
