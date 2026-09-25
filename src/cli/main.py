# src/cli/main.py

import argparse

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Transcribe Japanese Zoom recordings with mlx-whisper and translate to English "
            "using Ollama translategemma:12b."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("input", help="Input m4a/mp4/wav file")
    p.add_argument("--output-dir", help="Output directory. Defaults to input directory.")

    p.add_argument("--language", default="ja", help="Source language code")
    p.add_argument("--target-language", default="en", help="Target language code")

    p.add_argument(
        "--asr-backend",
        choices=["auto", "python", "cli"],
        default="auto",
        help="Use mlx-whisper Python API or mlx_whisper CLI.",
    )
    p.add_argument(
        "--asr-model",
        default="mlx-community/whisper-large-v3-mlx",
        help="MLX Whisper model repo/path. Example: mlx-community/whisper-large-v3-turbo",
    )
    p.add_argument(
        "--mlx-cli",
        default="mlx_whisper",
        help="mlx_whisper CLI executable name/path when using CLI backend.",
    )
    p.add_argument(
        "--initial-prompt",
        default="",
        help="Optional Whisper initial prompt with domain vocabulary or style hints.",
    )

    p.add_argument("--chunk-seconds", type=float, default=600.0)
    p.add_argument("--overlap-seconds", type=float, default=2.0)
    p.add_argument("--denoise", action="store_true", help="Apply conservative ffmpeg denoising.")

    p.add_argument("--translate-model", default="translategemma:12b")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument(
        "--num-predict",
        type=int,
        default=0,
        help="Max tokens to generate per translation request. 0 means leave unset.",
    )
    p.add_argument("--batch-segments", type=int, default=10)
    p.add_argument("--max-batch-chars", type=int, default=1200)
    p.add_argument("--keep-alive", default="10m")
    p.add_argument("--timeout", type=int, default=600)

    p.add_argument("--max-cue-duration", type=float, default=7.0)
    p.add_argument("--max-cue-chars", type=int, default=180)
    p.add_argument("--max-cue-gap", type=float, default=0.5)

    p.add_argument("--resume", action="store_true", help="Use cached ASR/translation segments if present.")
    p.add_argument("--force", action="store_true", help="Ignore caches and reprocess.")
    p.add_argument("--skip-translate", action="store_true", help="Only transcribe; do not translate.")
    p.add_argument("--keep-temp", action="store_true", help="Keep temporary audio chunks.")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()
