# src/output/generation.py

from dataclasses import asdict
import time
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from src.data.models import Segment

def ts_parts(ts: float) -> tuple[int, int, int, int]:
    total_ms = max(0, int(round(ts * 1000)))
    h = total_ms // 3_600_000
    m = (total_ms % 3_600_000) // 60_000
    s = (total_ms % 60_000) // 1_000
    ms = total_ms % 1_000
    return h, m, s, ms


def fmt_plain(ts: float) -> str:
    h, m, s, ms = ts_parts(ts)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def fmt_srt(ts: float) -> str:
    h, m, s, ms = ts_parts(ts)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_vtt(ts: float) -> str:
    h, m, s, ms = ts_parts(ts)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def build_cues(
    segments: List[Segment],
    max_duration: float,
    max_chars: int,
    max_gap: float,
) -> List[Dict[str, Any]]:
    cues: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for seg in segments:
        ja = seg.ja.strip()
        en = seg.en.strip()

        if not ja and not en:
            continue

        if current is None:
            current = {
                "start": seg.start,
                "end": seg.end,
                "ja": [ja],
                "en": [en],
                "chars": len(ja) + len(en),
            }
            continue

        gap = seg.start - current["end"]
        new_end = max(current["end"], seg.end)
        new_duration = new_end - current["start"]
        new_chars = current["chars"] + len(ja) + len(en)

        if gap <= max_gap and new_duration <= max_duration and new_chars <= max_chars:
            current["end"] = new_end
            current["ja"].append(ja)
            current["en"].append(en)
            current["chars"] = new_chars
        else:
            cues.append(current)
            current = {
                "start": seg.start,
                "end": seg.end,
                "ja": [ja],
                "en": [en],
                "chars": len(ja) + len(en),
            }

    if current is not None:
        cues.append(current)

    return cues


def cue_text(cue: Dict[str, Any]) -> str:
    ja = "".join([x for x in cue.get("ja", []) if x])
    en = " ".join([x for x in cue.get("en", []) if x])

    if ja and en:
        return f"{ja}\n{en}"

    return ja or en


def write_plain(path: Path, segments: List[Segment], attr: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for seg in segments:
            text = getattr(seg, attr).strip()
            if not text:
                continue
            f.write(f"[{fmt_plain(seg.start)} --> {fmt_plain(seg.end)}] {text}\n")


def write_srt(path: Path, cues: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, cue in enumerate(cues, 1):
            text = cue_text(cue).replace("\r", " ")
            f.write(f"{i}\n")
            f.write(f"{fmt_srt(cue['start'])} --> {fmt_srt(cue['end'])}\n")
            f.write(f"{text}\n\n")


def write_vtt(path: Path, cues: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, cue in enumerate(cues, 1):
            text = cue_text(cue).replace("-->", "->").replace("\r", " ")
            f.write(f"{i}\n")
            f.write(f"{fmt_vtt(cue['start'])} --> {fmt_vtt(cue['end'])}\n")
            f.write(f"{text}\n\n")


def write_json(
    path: Path,
    args: argparse.Namespace,
    input_path: Path,
    duration: float,
    segments: List[Segment],
    asr_backend_used: str,
) -> None:
    data = {
        "source": str(input_path),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_seconds": duration,
        "source_language": args.language,
        "target_language": args.target_language,
        "asr_backend": asr_backend_used,
        "asr_model": args.asr_model,
        "translation_model": args.translate_model,
        "settings": {
            "chunk_seconds": args.chunk_seconds,
            "overlap_seconds": args.overlap_seconds,
            "denoise": args.denoise,
            "temperature": args.temperature,
            "num_ctx": args.num_ctx,
            "batch_segments": args.batch_segments,
            "max_batch_chars": args.max_batch_chars,
            "initial_prompt": args.initial_prompt,
        },
        "segments": [asdict(s) for s in segments],
    }

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
