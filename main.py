#!/usr/bin/env python3
"""
Transcribe Japanese Zoom recordings with mlx-whisper, 
then translate to English using locally hosted Ollama translategemma:12b.

Outputs for input.mp4:
  input.ja.txt            timestamped Japanese transcript
  input.en.txt            timestamped English translation
  input.bilingual.srt     bilingual SRT: Japanese line + English line
  input.bilingual.vtt     bilingual WebVTT
  input.segments.json     structured segments with timestamps and ASR metadata

Example:
  python main.py meeting.mp4 \
      --asr-model mlx-community/whisper-large-v3-mlx \
      --translate-model translategemma:12b \
      --chunk-seconds 600 \
      --overlap-seconds 2
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional
import sys

from src.data.models import Segment
from src.cli.main import parse_args
from src.utils.utils import media_duration, require_binary, split_audio, extract_audio
from src.asr.backend import load_segments, determine_asr_backend, transcribe_chunk, normalize_asr_segments, sort_and_clean_segments, save_segments
from src.translation.client import OllamaClient, check_ollama
from src.translation.logic import translate_segments
from src.output.generation import write_plain, build_cues, write_srt, write_vtt, write_json

LOG = logging.getLogger("moji")

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        require_binary("ffmpeg")
        require_binary("ffprobe")

        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        outdir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else input_path.parent
        )
        outdir.mkdir(parents=True, exist_ok=True)

        stem = input_path.stem
        workdir = outdir / f"{stem}_work"
        workdir.mkdir(parents=True, exist_ok=True)

        asr_cache = workdir / "asr_segments.json"
        tr_cache = workdir / "translated_segments.json"

        duration = media_duration(input_path)

        segments: Optional[List[Segment]] = None
        asr_backend_used = "cached"

        if args.resume and tr_cache.exists() and not args.force:
            segments = load_segments(tr_cache)
            asr_backend_used = "cached-translated"
            LOG.info("Loaded cached translated segments from %s", tr_cache)

        elif args.resume and asr_cache.exists() and not args.force:
            segments = load_segments(asr_cache)
            asr_backend_used = "cached-asr"
            LOG.info("Loaded cached ASR segments from %s", asr_cache)

        else:
            backend = determine_asr_backend(args)
            asr_backend_used = backend
            LOG.info("Using ASR backend: %s", backend)

            wav = extract_audio(input_path, workdir, denoise=args.denoise)
            wav_duration = media_duration(wav)

            chunks = split_audio(
                wav,
                workdir,
                wav_duration,
                args.chunk_seconds,
                args.overlap_seconds,
            )

            LOG.info("Processing %d audio chunks", len(chunks))

            all_segments: List[Segment] = []
            next_id = 0

            for chunk in chunks:
                LOG.info(
                    "Transcribing chunk %d/%d nominal %.1f-%.1f s",
                    chunk["index"] + 1,
                    len(chunks),
                    chunk["nominal_start"],
                    chunk["nominal_end"],
                )

                result = transcribe_chunk(chunk["path"], backend, args)
                norm, next_id = normalize_asr_segments(result, chunk, next_id)
                all_segments.extend(norm)

            segments = sort_and_clean_segments(all_segments)
            save_segments(asr_cache, segments)
            LOG.info("Saved ASR cache: %s", asr_cache)

        if args.skip_translate:
            LOG.info("Skipping translation because --skip-translate was set.")
        else:
            client = OllamaClient(
                base_url=args.ollama_url,
                model=args.translate_model,
                timeout=args.timeout,
                keep_alive=args.keep_alive,
            )

            check_ollama(client, args.translate_model)

            if args.resume and tr_cache.exists() and not args.force:
                LOG.info("Using cached translations.")
            else:
                translate_segments(segments, client, args)
                save_segments(tr_cache, segments)
                LOG.info("Saved translation cache: %s", tr_cache)

        # Outputs
        ja_txt = outdir / f"{stem}.ja.txt"
        en_txt = outdir / f"{stem}.en.txt"
        srt_path = outdir / f"{stem}.bilingual.srt"
        vtt_path = outdir / f"{stem}.bilingual.vtt"
        json_path = outdir / f"{stem}.segments.json"

        write_plain(ja_txt, segments, "ja")
        write_plain(en_txt, segments, "en")

        cues = build_cues(
            segments,
            max_duration=args.max_cue_duration,
            max_chars=args.max_cue_chars,
            max_gap=args.max_cue_gap,
        )

        write_srt(srt_path, cues)
        write_vtt(vtt_path, cues)
        write_json(json_path, args, input_path, duration, segments, asr_backend_used)

        LOG.info("Japanese transcript: %s", ja_txt)
        LOG.info("English translation: %s", en_txt)
        LOG.info("Bilingual SRT:       %s", srt_path)
        LOG.info("Bilingual VTT:       %s", vtt_path)
        LOG.info("Structured JSON:     %s", json_path)

        if not args.keep_temp:
            for f in workdir.glob("chunk_*.wav"):
                f.unlink(missing_ok=True)

            wav_file = workdir / "audio_16k_mono.wav"
            if wav_file.exists():
                wav_file.unlink()

            LOG.info("Removed temporary audio files. Use --keep-temp to retain them.")

    except Exception as exc:
        LOG.exception("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
