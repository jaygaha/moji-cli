# src/translation/logic.py

import argparse
from src.translation.client import OllamaClient
from src.data.models import Segment
from src.translation.client import OllamaError
import json
import re
import logging
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)
TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional Japanese-to-English subtitle translator. "
    "Translate natural spoken Japanese into natural, readable English suitable for subtitles. "
    "Preserve meaning and tone. Do not add commentary. Do not omit segments. "
    "Return only valid JSON."
)


def parse_json_content(content: str) -> Dict[str, Any]:
    content = content.strip()

    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z0-9]*\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", content, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    m = re.search(r"\[.*\]", content, re.S)
    if m:
        try:
            return {"segments": json.loads(m.group(0))}
        except json.JSONDecodeError:
            pass

    raise OllamaError(f"Could not parse JSON from model response: {content[:500]}")

def translate_batch(
    batch: List[Segment],
    client: OllamaClient,
    args: argparse.Namespace,
) -> Dict[int, str]:
    payload_segments = [{"id": s.id, "ja": s.ja.strip()} for s in batch]

    schema_example = '{"segments":[{"id":1,"en":"English text"}]}'

    user_prompt = (
        f"""Translate the following Japanese subtitle segments to natural English.

                    Rules:
                    - Keep all segment ids unchanged.
                    - Do not omit any segment.
                    - Do not add explanations.
                    - Return only JSON matching this schema:
                        {schema_example}

                    Input JSON:
                    {json.dumps(payload_segments, ensure_ascii=False, indent=1)}
                """
    )

    messages = [
        {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    options: Dict[str, Any] = {
        "temperature": args.temperature,
        "num_ctx": args.num_ctx,
    }

    if args.num_predict > 0:
        options["num_predict"] = args.num_predict

    resp = client.chat(messages, options=options, json_format=True)
    content = (resp.get("message") or {}).get("content", "")

    parsed = parse_json_content(content)

    if isinstance(parsed, list):
        parsed = {"segments": parsed}

    if not isinstance(parsed, dict) or "segments" not in parsed:
        raise OllamaError(f"Unexpected translation response: {content[:500]}")

    translated: Dict[int, str] = {}

    for item in parsed.get("segments", []):
        if not isinstance(item, dict):
            continue

        try:
            sid = int(item.get("id"))
        except Exception:
            continue

        en = (
            item.get("en")
            or item.get("english")
            or item.get("translation")
            or ""
        )
        en = str(en).strip()

        if en:
            translated[sid] = en

    missing = [s.id for s in batch if s.id not in translated]
    if missing:
        raise OllamaError(f"Missing translations for segment ids: {missing}")

    return translated

def split_long_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    current = ""

    pieces = re.split(r"(?<=[。！？!?\n])", text)

    for piece in pieces:
        if not piece:
            continue

        if len(current) + len(piece) <= max_chars:
            current += piece
        else:
            if current:
                parts.append(current)
                current = ""

            while len(piece) > max_chars:
                parts.append(piece[:max_chars])
                piece = piece[max_chars:]

            current = piece

    if current:
        parts.append(current)

    return [p.strip() for p in parts if p.strip()]

def translate_long_segment(
    seg: Segment,
    client: OllamaClient,
    args: argparse.Namespace,
) -> str:
    pieces = split_long_text(seg.ja.strip(), max(200, args.max_batch_chars))
    translations: List[str] = []

    for piece in pieces:
        fake = Segment(id=0, start=seg.start, end=seg.end, ja=piece)
        out = translate_batch([fake], client, args)
        translations.append(out[0])

    return " ".join(translations).strip()

def translate_segments(
    segments: List[Segment],
    client: OllamaClient,
    args: argparse.Namespace,
) -> None:
    pending = [s for s in segments if not s.en.strip()]
    if not pending:
        LOG.info("All segments already translated.")
        return

    current_batch_segments = max(1, args.batch_segments)
    i = 0

    while i < len(pending):
        seg = pending[i]

        # Very long segments are translated piecewise.
        if len(seg.ja) > int(args.max_batch_chars * 1.5):
            try:
                seg.en = translate_long_segment(seg, client, args)
            except Exception as exc:
                LOG.error("Failed to translate long segment %d: %s", seg.id, exc)
                seg.en = ""
            i += 1
            continue

        batch: List[Segment] = []
        chars = 0

        while i < len(pending) and len(batch) < current_batch_segments:
            cand = pending[i]
            add_chars = len(cand.ja)

            if batch and chars + add_chars > args.max_batch_chars:
                break

            batch.append(cand)
            chars += add_chars
            i += 1

            if chars >= args.max_batch_chars:
                break

        if not batch:
            i += 1
            continue

        try:
            translated = translate_batch(batch, client, args)
            for item in batch:
                item.en = translated.get(item.id, "")

            # Gradually restore batch size after success.
            if current_batch_segments < args.batch_segments:
                current_batch_segments = min(
                    args.batch_segments,
                    current_batch_segments + 1,
                )

        except Exception as exc:
            if len(batch) > 1:
                current_batch_segments = max(1, len(batch) // 2)
                i -= len(batch)
                LOG.warning(
                    "Translation batch failed: %s. Reducing batch size to %d and retrying.",
                    exc,
                    current_batch_segments,
                )
            else:
                LOG.error("Translation failed for segment %d: %s", batch[0].id, exc)
                batch[0].en = ""
                i += 1