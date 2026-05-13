from __future__ import annotations

"""Container-side transcription entrypoint.

This module supports two input modes:
- URL mode (`--url`): download audio with yt-dlp to temporary storage
- Local file mode (`--input-file`): extract mono 16 kHz WAV with ffmpeg

Then it runs faster-whisper and writes a TXT transcript to the requested path.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yt_dlp
from faster_whisper import WhisperModel
from tqdm import tqdm


_VALID_COMPUTE_TYPES: frozenset[str] = frozenset(
    {"int8", "float32", "float16", "int8_float16"}
)
_LANGUAGE_RE = re.compile(r"^[a-zA-Z]{2,8}(-[a-zA-Z0-9]{1,8})*$")

# Extensions considered to be media files when scanning a download directory
_MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".mp4", ".m4a", ".webm", ".ogg", ".wav", ".flac",
     ".opus", ".mkv", ".avi", ".mov", ".aac", ".wma", ".m4v"}
)


def safe_filename(value: str, max_len: int = 180) -> str:
    """Return a filesystem-safe stem for output file names.

    :param value: Raw file name candidate.
    :param max_len: Maximum allowed output length.
    :returns: Sanitized, non-empty file name stem.
    """
    value = re.sub(r'[\x00-\x1f\x7f-\x9f]', "", value)  # strip control characters
    value = re.sub(r'[\\/*?:"<>|]', "_", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:max_len] or "transcript"


def warn_overwrite(path: Path) -> None:
    """Print a standardized warning when an existing output file will be replaced."""
    print(f"WARN: overwriting existing output file: {path}", file=sys.stderr, flush=True)


def run_cmd(cmd: list[str], timeout_seconds: int = 1800) -> str:
    """Run a subprocess command and return stripped stdout.

    :param cmd: Command and arguments to execute.
    :param timeout_seconds: Max execution time in seconds.
    :returns: Command stdout with trailing whitespace stripped.
    :raises RuntimeError: If command exits with non-zero status.
    :raises subprocess.TimeoutExpired: If command exceeds timeout.
    """
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDERR:\n"
            + result.stderr
        )

    return result.stdout.strip()


def create_progress_bar(
    description: str,
    total: float | None,
    unit: str,
    unit_scale: bool = False,
) -> tqdm:
    """Create a consistent progress bar with elapsed/remaining information.

    :param description: Human-readable step label.
    :param total: Total expected work units, or ``None`` if unknown.
    :param unit: Unit suffix used by tqdm (e.g. ``"B"``, ``"s"``).
    :param unit_scale: Whether to use SI scaling for displayed values.
    :returns: Configured tqdm progress bar.
    """
    return tqdm(
        total=total,
        desc=description,
        unit=unit,
        unit_scale=unit_scale,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )


def yt_dlp_options_base(cookies: str | None) -> dict:
    """Build shared yt-dlp options for embedded Python API calls.

    :param cookies: Optional path to cookies file for restricted media.
    :returns: Dictionary of yt-dlp options.
    """
    options: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if cookies:
        options["cookiefile"] = cookies
    return options


def download_url_media(
    url: str,
    tmp_dir: Path,
    cookies: str | None,
) -> tuple[Path, str]:
    """Download URL media to *tmp_dir* and return the file path plus a safe output stem.

    The output stem is derived from the media title and ID so callers can
    construct a unique output file name without a second yt-dlp metadata call.

    :param url: Media URL supported by yt-dlp.
    :param tmp_dir: Temporary directory for downloaded media.
    :param cookies: Optional path to cookies file for yt-dlp.
    :returns: ``(downloaded_media_path, safe_filename_stem)``
    :raises RuntimeError: If download fails or no output file is found.
    """
    progress = create_progress_bar("Downloading media", total=None, unit="B", unit_scale=True)

    def progress_hook(status: dict) -> None:
        state = status.get("status")
        if state == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            downloaded = status.get("downloaded_bytes") or 0
            if total and progress.total is None:
                progress.total = float(total)
            if downloaded > progress.n:
                progress.update(downloaded - progress.n)
        elif state == "finished":
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            if total:
                if progress.total is None:
                    progress.total = float(total)
                if total > progress.n:
                    progress.update(total - progress.n)

    options = yt_dlp_options_base(cookies)
    options.update(
        {
            "format": "bestaudio/best",
            "paths": {"home": str(tmp_dir)},
            "outtmpl": {"default": "%(id)s.%(ext)s"},
            "progress_hooks": [progress_hook],
        }
    )

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            # Capture the yt-dlp-generated path before closing the context
            ydl_expected_path = Path(ydl.prepare_filename(info)).resolve()
    except yt_dlp.DownloadError as exc:
        raise RuntimeError(f"Failed to download media from '{url}': {exc}") from exc
    finally:
        progress.close()

    # Build a safe output stem from the single metadata call
    title = f"{info.get('title', 'media')} [{info.get('id', 'unknown')}]"
    stem = safe_filename(title)

    # Prefer the path yt-dlp itself recorded in requested_downloads
    downloads = info.get("requested_downloads") or []
    if downloads:
        filepath = downloads[0].get("filepath", "")
        if filepath:
            candidate = Path(filepath).resolve()
            if candidate.is_file():
                return candidate, stem

    # Secondary fallback: the path yt-dlp would have used
    if ydl_expected_path.is_file():
        return ydl_expected_path, stem

    # Last-resort fallback: scan tmp_dir for any media file
    media_files = sorted(
        p.resolve()
        for p in tmp_dir.glob("*")
        if p.is_file() and p.suffix.lower() in _MEDIA_EXTENSIONS
    )
    if media_files:
        return media_files[0], stem

    raise RuntimeError("Downloaded media file was not found in temporary directory.")


def get_media_duration_seconds(input_path: Path) -> float | None:
    """Return media duration in seconds using ffprobe.

    :param input_path: Path to media file.
    :returns: Duration in seconds, or ``None`` when unavailable.
    """
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-loglevel",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    output = run_cmd(cmd)
    value = output.strip()
    if not value or value.lower() == "n/a":
        return None
    try:
        duration = float(value)
    except ValueError:
        return None
    return duration if duration > 0 else None


def probe_media_kind(input_path: Path) -> str:
    """Classify media streams in ``input_path`` using ffprobe.

    Returns one of:
    - "video_with_audio"
    - "audio"
    - "video_without_audio"
    - "unknown"

    :param input_path: Path to local media file.
    :returns: Media kind classification string.
    :raises RuntimeError: If ffprobe execution fails.
    """
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-loglevel",
        "error",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]

    output = run_cmd(cmd)
    stream_types = {line.strip() for line in output.splitlines() if line.strip()}

    has_audio = "audio" in stream_types
    has_video = "video" in stream_types

    if has_video and has_audio:
        return "video_with_audio"
    if has_audio:
        return "audio"
    if has_video:
        return "video_without_audio"
    return "unknown"


def convert_local_media_to_tmp_audio(
    input_path: Path,
    tmp_dir: Path,
    timeout_seconds: int = 3600,
) -> Path:
    """Convert local media file to mono 16 kHz WAV in ``tmp_dir``.

    :param input_path: Source media file path.
    :param tmp_dir: Temporary directory for converted WAV output.
    :param timeout_seconds: Maximum time allowed for the conversion.
    :returns: Path to generated WAV file.
    :raises RuntimeError: If no usable audio stream exists or conversion fails.
    """
    media_kind = probe_media_kind(input_path)

    if media_kind == "video_without_audio":
        raise RuntimeError(f"Input file has video stream but no audio stream: {input_path}")
    if media_kind == "unknown":
        raise RuntimeError(f"Input file has no detectable audio stream: {input_path}")

    output_path = tmp_dir / "audio.wav"
    try:
        duration = get_media_duration_seconds(input_path)
    except Exception:
        duration = None
    progress = create_progress_bar("Extracting audio", total=duration, unit="s")

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-progress",
        "pipe:1",
        "-nostats",
        str(output_path),
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        try:
            process.kill()
        except Exception:
            pass
        raise RuntimeError(
            "FFmpeg process failed to start properly (missing stdout pipe). "
            "Check that FFmpeg is available in the container and the input file is readable."
        )

    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            if line.startswith("out_time_ms="):
                out_time_ms = line.split("=", 1)[1]
                if out_time_ms.isdigit():
                    seconds = int(out_time_ms) / 1_000_000
                    if progress.total is None:
                        progress.update(max(0.0, seconds - progress.n))
                    elif seconds > progress.n:
                        progress.update(seconds - progress.n)
            elif line == "progress=end":
                break

        try:
            stdout_remaining, stderr_remaining = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise RuntimeError(
                f"FFmpeg audio extraction timed out after {timeout_seconds}s. "
                f"The input file may be too large or corrupted: {input_path}"
            )
    finally:
        progress.close()

    if process.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDERR:\n"
            + (stderr_remaining or "")
            + ("\nSTDOUT:\n" + stdout_remaining if stdout_remaining else "")
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg conversion finished but output WAV was not created.")

    return output_path


def format_time(seconds: float) -> str:
    """Convert seconds offset to ``HH:MM:SS``.

    :param seconds: Time offset in seconds.
    :returns: Formatted timestamp string.
    """
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def transcribe_audio(
    model: WhisperModel,
    audio_path: Path,
    output_path: Path,
    source_label: str,
    language: str | None,
    timestamps: bool,
) -> None:
    """Run faster-whisper on ``audio_path`` and write transcript.

    The output starts with source/language metadata followed by either:
    - timestamped lines: ``[HH:MM:SS - HH:MM:SS] text``
    - plain text lines (if ``timestamps`` is ``False``)

    :param model: Initialized faster-whisper model instance.
    :param audio_path: Path to input audio file.
    :param output_path: Destination TXT path.
    :param source_label: Human-readable source identifier written to output.
    :param language: Optional language code override (for example ``"en"``).
    :param timestamps: Whether to include segment timestamps.
    :returns: ``None``.
    :raises OSError: If output file cannot be created or written.
    :raises RuntimeError: If model transcription fails.
    """
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
    )
    try:
        duration = get_media_duration_seconds(audio_path)
    except Exception:
        duration = None
    progress = create_progress_bar("Transcribing", total=duration, unit="s")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_path.open("w", encoding="utf-8") as file:
            file.write(f"Source: {source_label}\n")
            file.write(f"Detected language: {info.language}\n")
            file.write(f"Language probability: {info.language_probability:.2f}\n")
            file.write("\n")

            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue

                if timestamps:
                    start = format_time(segment.start)
                    end = format_time(segment.end)
                    file.write(f"[{start} - {end}] {text}\n")
                else:
                    file.write(text + "\n")

                marker = max(float(segment.end), float(segment.start))
                if progress.total is None:
                    progress.update(max(0.0, marker - progress.n))
                elif marker > progress.n:
                    progress.update(marker - progress.n)
    finally:
        progress.close()


def process_url(
    model: WhisperModel,
    url: str,
    output_dir: Path,
    output_file: Path | None,
    language: str | None,
    timestamps: bool,
    cookies: str | None,
) -> None:
    """Handle URL input end-to-end: download, transcribe, save.

    The output file name is resolved from the media metadata returned by
    yt-dlp during the single download call, so no separate metadata query
    is made.

    :param model: Initialized faster-whisper model instance.
    :param url: URL supported by yt-dlp.
    :param output_dir: Output directory for auto-generated file names.
    :param output_file: Optional explicit output TXT path.
    :param language: Optional language code override.
    :param timestamps: Whether to include segment timestamps.
    :param cookies: Optional path to yt-dlp cookies file.
    :returns: ``None``.
    :raises RuntimeError: If download/transcription pipeline fails.
    :raises OSError: If output cannot be written.
    """
    # Check output directory is writable before the potentially long download
    check_dir = output_file.parent if output_file else output_dir
    check_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(check_dir, os.W_OK):
        raise OSError(f"Output directory is not writable: {check_dir}")

    print("Input type: URL", flush=True)
    print(f"Input: {url}", flush=True)

    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_name:
        tmp_dir = Path(tmp_name)

        media_path, stem = download_url_media(url=url, tmp_dir=tmp_dir, cookies=cookies)
        output_path = output_file or (output_dir / f"{stem}.txt")

        if output_path.exists():
            warn_overwrite(output_path)

        print(f"Output: {output_path}", flush=True)

        audio_path = convert_local_media_to_tmp_audio(input_path=media_path, tmp_dir=tmp_dir)

        transcribe_audio(
            model=model,
            audio_path=audio_path,
            output_path=output_path,
            source_label=url,
            language=language,
            timestamps=timestamps,
        )

    print(f"Saved: {output_path}", flush=True)


def process_local_file(
    model: WhisperModel,
    input_file: Path,
    output_dir: Path,
    output_file: Path | None,
    language: str | None,
    timestamps: bool,
) -> None:
    """Handle local media input end-to-end: extract audio, transcribe, save.

    :param model: Initialized faster-whisper model instance.
    :param input_file: Local audio/video path inside container.
    :param output_dir: Output directory for auto-generated file names.
    :param output_file: Optional explicit output TXT path.
    :param language: Optional language code override.
    :param timestamps: Whether to include segment timestamps.
    :returns: ``None``.
    :raises FileNotFoundError: If input file does not exist.
    :raises RuntimeError: If input is invalid or extraction/transcription fails.
    :raises OSError: If output cannot be written.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not input_file.is_file():
        raise RuntimeError(f"Input path is not a file: {input_file}")

    output_path = output_file or (output_dir / f"{safe_filename(input_file.stem)}.txt")

    # Check output directory is writable before any processing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(output_path.parent, os.W_OK):
        raise OSError(f"Output directory is not writable: {output_path.parent}")

    if output_path.exists():
        warn_overwrite(output_path)

    print("Input type: local media file", flush=True)
    print(f"Input: {input_file}", flush=True)
    print(f"Output: {output_path}", flush=True)

    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_name:
        tmp_dir = Path(tmp_name)

        audio_path = convert_local_media_to_tmp_audio(input_path=input_file, tmp_dir=tmp_dir)
        transcribe_audio(
            model=model,
            audio_path=audio_path,
            output_path=output_path,
            source_label=str(input_file),
            language=language,
            timestamps=timestamps,
        )

    print(f"Saved: {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    """Define and parse container CLI arguments.

    :returns: Parsed CLI arguments namespace.
    :raises SystemExit: If user passes invalid CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe URL, local audio, or local video to TXT inside Docker."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--url", help="Any URL supported by yt-dlp.")
    input_group.add_argument(
        "--input-file", help="Local audio or video file mounted inside the container."
    )

    parser.add_argument(
        "--output-dir",
        default="/out",
        help="Directory where TXT files are written. Default: /out.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Exact output TXT file path. Valid only for single input.",
    )

    parser.add_argument(
        "--model",
        default="medium",
        help="Whisper model, e.g. small, medium, large-v3. Default: medium.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code, e.g. pl or en. Empty means autodetect.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device: cpu or cuda. Default: cpu.",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Compute type. CPU commonly int8/float32, CUDA commonly float16.",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Write plain text without timestamps.",
    )
    parser.add_argument(
        "--cookies",
        default=None,
        help="Optional cookies.txt file for yt-dlp.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations not expressible in argparse metadata.

    :param args: Parsed CLI arguments.
    :returns: ``None``.
    :raises SystemExit: If unsupported argument combinations are used.
    """
    if args.output_file and not str(args.output_file).lower().endswith(".txt"):
        raise SystemExit("--output-file must end with .txt")

    if args.compute_type not in _VALID_COMPUTE_TYPES:
        raise SystemExit(
            f"--compute-type '{args.compute_type}' is not valid. "
            f"Allowed values: {', '.join(sorted(_VALID_COMPUTE_TYPES))}"
        )

    if args.device == "cpu" and args.compute_type == "float16":
        raise SystemExit("--compute-type float16 is not valid for CPU. Use int8 or float32.")

    if args.language and not _LANGUAGE_RE.match(args.language):
        raise SystemExit(
            f"--language '{args.language}' does not look like a valid language code. "
            "Use language codes such as: en, pl, de, zh-CN, pt-BR."
        )


def env_bool(name: str, default: bool) -> bool:
    """Parse boolean value from environment variable.

    :param name: Environment variable name.
    :param default: Value returned if variable is unset.
    :returns: Parsed boolean value.
    """
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def main() -> None:
    """Program entrypoint for a single URL or local input transcription run.

    :returns: ``None``.
    :raises SystemExit: On any error — always prints a clean message instead of
        a raw traceback so the user knows what went wrong.
    """
    args = parse_args()
    validate_args(args)

    language = args.language.strip() if args.language else None
    language = language or None

    output_dir = Path(args.output_dir)
    output_file = Path(args.output_file) if args.output_file else None

    print(f"Loading Whisper model: {args.model}", flush=True)
    download_root = os.getenv("WHISPER_DOWNLOAD_ROOT", "/opt/faster-whisper-cache")
    local_files_only = env_bool("WHISPER_LOCAL_FILES_ONLY", True)

    try:
        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            download_root=download_root,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        if local_files_only:
            raise SystemExit(
                "Requested model is not available in the preloaded image cache.\n"
                f"Model: {args.model}\n"
                "Rebuild the image with model preloading enabled."
            ) from exc
        raise

    try:
        if args.url:
            process_url(
                model=model,
                url=args.url,
                output_dir=output_dir,
                output_file=output_file,
                language=language,
                timestamps=not args.no_timestamps,
                cookies=args.cookies,
            )
        else:
            process_local_file(
                model=model,
                input_file=Path(args.input_file),
                output_dir=output_dir,
                output_file=output_file,
                language=language,
                timestamps=not args.no_timestamps,
            )
    except SystemExit:
        raise
    except KeyboardInterrupt:
        raise SystemExit("\nInterrupted.")
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
