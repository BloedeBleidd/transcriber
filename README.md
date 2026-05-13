# Universal Local Transcriber

A Docker-based CLI for transcribing:

- URLs supported by `yt-dlp`
- local audio files
- local video files
- batch lists (`.txt`, `.list`, `.urls`)

Output is plain `.txt` transcript files, with or without timestamps.

## Why this project

This project gives one stable entrypoint:

```bash
./transcribe.sh <input> <output>
```

`<input>` can be URL/file/list. `<output>` can be exact `.txt` file or a directory.

## How it works

1. `transcribe.sh` validates arguments, resolves defaults, prints effective configuration, and builds or reuses the Docker image.
2. During image build, selected `faster-whisper` model variants are preloaded into the image cache. By default, only the `medium` model is preloaded.
3. At runtime, model loading is restricted to local preloaded files (no model downloads).
4. `transcribe.sh` runs a short-lived container (`app/transcriber.py`) with:
   - `yt-dlp` for URL media acquisition
   - `ffmpeg/ffprobe` for media probing/extraction
   - `faster-whisper` for transcription
5. Progress bars are shown for downloading, audio extraction and transcription (with elapsed and ETA when total is known).
6. Temporary media files are stored only in container `/tmp` mounted as tmpfs.
7. Preloaded Whisper models are stored inside the Docker image layers (persisting on the host in Docker's internal storage).
8. Only final `.txt` files are written to host output paths by the transcription process.
9. Input files and cookies are mounted as **file-level** bind mounts — only the specific file is visible inside the container, not its parent directory.
10. In batch mode all items are attempted; per-item failures are collected and reported at the end instead of aborting the whole run.

## Requirements

- **Docker** (any recent version)
- **Bash 4+** on the host

Everything else (Python, ffmpeg, yt-dlp, faster-whisper, …) runs inside the Docker container.

## Quick start

### 1) Run transcription

The script automatically builds the required Docker image on the first run.

```bash
./transcribe.sh "https://youtu.be/dQw4w9WgXcQ" ./youtube_en.txt --language en
```

Manual build (optional):

```bash
docker build --build-arg PRELOAD_MODELS=medium -t transcriber:medium .
```

### 2) Usage examples

```bash
./transcribe.sh "https://example.com/podcast-episode" ./podcast_es.txt --language es
./transcribe.sh ./recording_pl.wav ./recording_pl.txt --language pl --no-timestamps
./transcribe.sh ./meeting_en.mp4 ./meeting_en.txt --language en --device cpu
./transcribe.sh ./conferencia_es.mp4 ./transcripts --language es --device cuda --compute-type float16
./transcribe.sh ./inputs.txt ./transcripts --language pl
./transcribe.sh "https://example.com/restricted-video" ./restricted.txt --cookies ./cookies.txt
./transcribe.sh ./demo.mp4 ./demo.txt --build-mode fresh
```

## Batch file format

Example `inputs.txt`:

```txt
https://youtu.be/dQw4w9WgXcQ
https://example.com/podcast-episode
./local-audio-pl.wav
./local-video-en.mp4
# comment lines are ignored
```

For batch mode, relative paths are resolved against the location of `inputs.txt`.

Note: Batch mode currently starts one short-lived container per input item. This keeps path handling simple, but repeated model loading makes large batches slower.

## CLI reference

Show help:

```bash
./transcribe.sh --help
```

Key options:

- `--model <name>` (default: `medium`)
- `--language <code>` (default: auto-detect)
- `--no-timestamps`
- `--cookies <path>`
- `--device <cpu|cuda>` (default: `cpu`)
- `--compute-type <type>`
- `--tmp-size <size>`
- `--image-prefix <name>` (default: `transcriber`)
- `--build-mode <auto|rebuild|fresh>`

## CPU/GPU selection

```bash
./transcribe.sh ./meeting_en.mp4 ./meeting_en.txt --language en --device cpu
./transcribe.sh ./conferencia_es.mp4 ./transcripts --language es --device cuda --compute-type float16
```

Notes:

- The default Dockerfile is CPU-oriented.
- `--device cuda` requires Docker GPU support (`--gpus all`) and NVIDIA Container Toolkit on host.
- GPU mode requires an image/runtime that provides CUDA 12 and cuDNN 9 libraries. If the default image fails with `--device cuda`, use `cpu` or provide a dedicated CUDA-enabled Docker image.
- Models are preloaded into the Docker image at build time.
- Runtime uses local preloaded model files only.

## Build mode

`--build-mode` controls how Docker image build is handled:

- `auto` (default): build only if image is missing
- `rebuild`: always rebuild using normal Docker cache
- `fresh`: remove local image and rebuild with `--no-cache --pull`

## Version pinning

The project pins runtime versions in:

- [`Dockerfile`](./Dockerfile)
- [`requirements.txt`](./requirements.txt)

Pinned items include:

- base image tag: `python:3.12.12-slim-bookworm`
- Debian package: `ffmpeg` (stable from Debian bookworm)
- Python dependencies including `yt-dlp`, `faster-whisper` and direct `faster-whisper` runtime deps

## Licensing and compatibility (checked on May 11, 2026)

### Project license used here

This repository is licensed as **GPL-3.0-or-later**.

Rationale: GPL-3.0-or-later is chosen to keep redistribution of the complete Docker-based project conservative with respect to GPL-family components included in the image.

`LICENSE` contains the full GNU GPL v3 text, with an explicit `or later` notice for this project.

### Dependency license map

| Component | Version source | License |
|---|---|---|
| project code (`transcribe.sh`, `app/transcriber.py`) | this repo | GPL-3.0-or-later |
| Python runtime | Docker image (`python:3.12.12-slim-bookworm`) | PSF-2.0 (+ bundled software licenses) |
| ffmpeg (Debian package) | Debian bookworm repository version used at build time | GPL/LGPL mix; Debian package notes GPL-family result |
| yt-dlp (PyPI wheel/sdist) | `2026.3.17` | Unlicense |
| faster-whisper | `1.2.1` | MIT |
| ctranslate2 | `4.7.1` | MIT |
| huggingface-hub | `1.14.0` | Apache-2.0 |
| tokenizers | `0.23.1` | Apache-2.0 |
| onnxruntime | `1.25.1` | MIT |
| av (PyAV) | `17.0.1` | BSD-3-Clause |
| tqdm | `4.67.3` | MPL-2.0 + MIT parts |

### Primary sources

- FFmpeg legal/licensing: https://ffmpeg.org/legal.html
- Debian ffmpeg copyright metadata (bookworm): https://sources.debian.org/src/ffmpeg/7%3A5.1.8-0%2Bdeb12u1/debian/copyright
- yt-dlp PyPI licensing note: https://pypi.org/project/yt-dlp/
- faster-whisper license: https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE
- CTranslate2 license: https://github.com/OpenNMT/CTranslate2
- huggingface_hub license: https://github.com/huggingface/huggingface_hub/blob/main/LICENSE
- tokenizers license: https://github.com/huggingface/tokenizers
- onnxruntime license: https://github.com/microsoft/onnxruntime/blob/main/LICENSE
- PyAV license metadata: https://pypi.org/project/av/
- tqdm license page: https://tqdm.github.io/licence/
- Python license: https://docs.python.org/3.12/license.html

## Security and operational notes

- Container runs as current host UID/GID (when available).
- Input files are mounted read-only.
- Cookies file is mounted read-only and only when requested.
- Temporary media files are stored in container tmpfs (`/tmp`) and are deleted with the container.
- Preloaded Whisper models are stored inside the Docker image layers.
- The transcription runtime writes only requested output `.txt` files to host mounts.
- Docker still maintains its own image/layer storage (including preloaded models) in its internal storage on the host.

## Limitations

- GPU mode requires Docker GPU runtime (`--gpus all`) and compatible NVIDIA stack.
- URL transcription depends on `yt-dlp` extractor support and site restrictions.
