# moji (文字起こし Transcriber)

A CLI pipeline designed to transcribe Japanese audio/video recordings (Zoom meetings, interviews, lectures) with Apple Silicon hardware acceleration and translate them into English using local LLMs.

Powered by **Apple MLX Whisper** for speech-to-text and **Ollama (`translategemma:12b`)** for translation, `moji` runs 100% locally with zero cloud dependencies and complete data privacy.

---

## Key Features

- **Apple Silicon Acceleration**: Native MLX execution (`mlx-whisper`) leveraging the Neural Engine and unified memory on M-series chips.
- **Local Neural Translation**: Context-aware Japanese-to-English translation powered by Ollama (`translategemma:12b`).
- **Long Recording Handling**: Automatic audio extraction, optional denoising, and chunked processing (with configurable overlap) to handle multi-hour recordings without memory spikes or drift.
- **Rich Bilingual Outputs**: Automatically generates transcripts and subtitles for every recording:
  - `.ja.txt`: Timestamped Japanese transcript
  - `.en.txt`: Timestamped English translation
  - `.bilingual.srt`: Bilingual subtitle file (Japanese top line, English bottom line)
  - `.bilingual.vtt`: Bilingual WebVTT subtitle file
  - `.segments.json`: Structured JSON containing cue timings, confidence metrics, and metadata
- **Checkpoint & Resume**: Caches intermediate transcription (`asr_segments.json`) and translation (`translated_segments.json`). If interrupted or if output options need adjusting, use `--resume` to pick up instantly without reprocessing.
- **100% Offline & Private**: Audio and transcriptions never leave your local machine.

---

## Prerequisites

1. **Apple Silicon Mac**
2. **Python 3.10+**
3. **FFmpeg**: Required for audio extraction and chunking:
   ```bash
   brew install ffmpeg
   ```
4. **Ollama**: Required for translation:
   - [Install Ollama](https://ollama.com/)
   - Pull the translation model:
     ```bash
     ollama pull translategemma:12b
     ```

---

## Installation

1. Clone or navigate into the repository:
   ```bash
   cd moji-cli
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Basic Usage

Transcribe and translate a media file (`.mp4`, `.m4a`, `.wav`, `.mkv`, etc.):

```bash
python main.py "/path/to/meeting.mp4"
```

All outputs will be saved in the same directory as the input file (or specify `--output-dir`).

### Resume from Checkpoint

If a run was stopped, or you want to regenerate subtitles without re-transcribing or re-translating:

```bash
python main.py --resume "/path/to/meeting.mp4"
```

### Transcription Only (Skip Translation)

To only generate Japanese transcripts and subtitles without running Ollama translation:

```bash
python main.py --skip-translate "/path/to/meeting.mp4"
```

---

## Advanced Options & Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `input` | *(Required)* | Path to input audio/video file |
| `--output-dir` | Input file directory | Destination directory for output files |
| `--resume` | `False` | Resume using cached ASR/translation checkpoints if available |
| `--force` | `False` | Ignore existing caches and force a fresh run |
| `--skip-translate` | `False` | Skip translation step entirely |
| `--asr-model` | `mlx-community/whisper-large-v3-mlx` | Hugging Face repo ID or local path for MLX Whisper model |
| `--asr-backend` | `auto` | Backend for transcription: `auto`, `python`, or `cli` |
| `--initial-prompt` | `""` | Optional prompt to guide Whisper with specialized terminology |
| `--translate-model` | `translategemma:12b` | Ollama model name for translation |
| `--ollama-url` | `http://127.0.0.1:11434` | Base URL for Ollama API |
| `--chunk-seconds` | `600.0` (10 min) | Duration of audio chunks in seconds |
| `--overlap-seconds` | `2.0` | Overlap between chunks to prevent clipping sentences |
| `--denoise` | `False` | Apply conservative FFmpeg audio noise filter before transcription |
| `--batch-segments` | `10` | Number of segments sent per Ollama translation batch |
| `--max-cue-duration` | `7.0` | Maximum duration (seconds) for subtitle cues |
| `--keep-temp` | `False` | Keep temporary chunk `.wav` files after processing |
| `--verbose` | `False` | Enable debug logging output |

### Examples

**Using Whisper Turbo for faster transcription:**
```bash
python main.py meeting.mp4 --asr-model mlx-community/whisper-large-v3-turbo
```

**Providing domain vocabulary hints to Whisper:**
```bash
python main.py lecture.mp4 --initial-prompt "奈良医科大学, 笠原, 臨床研究, 症例報告"
```

**Custom chunking and denoising noisy audio:**
```bash
python main.py noisy_interview.m4a --denoise --chunk-seconds 300
```

---

## Output Files

For an input file named `recording.mp4`, the following files are produced:

```text
recording/
├── recording.ja.txt            # Clean Japanese text with timestamps
├── recording.en.txt            # English translation text with timestamps
├── recording.bilingual.srt     # Subtitles with Japanese and English side-by-side
├── recording.bilingual.vtt     # WebVTT format for browser players
└── recording.segments.json     # Full segment data, word timings, and quality metrics
```

Intermediate artifacts are kept in `recording_work/` for recovery:
- `asr_segments.json`: Cached ASR results
- `translated_segments.json`: Cached translation results

---

## License

MIT License. See [LICENSE](LICENSE) for details.

happy coding 👾
