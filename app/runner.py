#!/usr/bin/env python3
"""Host-side runner: validates arguments, builds the Docker image, launches the container.

This module replaces the heavy argument-parsing and Docker-orchestration logic
previously embedded in ``transcribe.sh``.  The shell script is now a thin
wrapper that checks for ``python3`` and delegates here via ``exec``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

# Repository root: one directory above this file (which lives in ``app/``)
_REPO_ROOT = Path(__file__).resolve().parent.parent

_VALID_COMPUTE_TYPES: frozenset[str] = frozenset(
    {"auto", "int8", "float32", "float16", "int8_float16"}
)
_VALID_DEVICES: frozenset[str] = frozenset({"cpu", "cuda"})
_VALID_BUILD_MODES: frozenset[str] = frozenset({"auto", "rebuild", "fresh"})
_LIST_EXTENSIONS: frozenset[str] = frozenset({".txt", ".list", ".urls"})

_URL_RE = re.compile(r"^https?://")
_MODEL_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
_IMAGE_PREFIX_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9.:/\-_]*$")
_TMP_SIZE_RE = re.compile(r"^\d+[bBkKmMgGtT]?$")
_LANGUAGE_RE = re.compile(r"^[a-zA-Z]{2,8}(-[a-zA-Z0-9]{1,8})*$")

_DEFAULT_MODEL = "medium"
_DEFAULT_DEVICE = "cpu"
_DEFAULT_COMPUTE_TYPE = "auto"
_DEFAULT_TMP_SIZE = "8g"
_DEFAULT_IMAGE_PREFIX = "transcriber"
_DEFAULT_BUILD_MODE = "auto"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _die(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _warn(message: str) -> None:
    print(f"WARN: {message}", file=sys.stderr)


def _is_url(value: str) -> bool:
    return bool(_URL_RE.match(value))


def _is_list_file(path: Path) -> bool:
    return path.suffix.lower() in _LIST_EXTENSIONS


def _is_txt_file(path: Path) -> bool:
    return path.suffix.lower() == ".txt"


def _mkdir_or_raise(path: Path) -> None:
    """Create *path* (and parents) or raise RuntimeError on failure."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot create directory '{path}': {exc}") from exc


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcribe.sh",
        description="Universal local transcription runner (Docker-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Inputs:
  1) URL supported by yt-dlp
       ./transcribe.sh "https://youtu.be/..." ./transcript.txt

  2) Local audio file
       ./transcribe.sh ./audio.mp3 ./transcript.txt

  3) Local video file
       ./transcribe.sh ./video.mp4 ./transcript.txt

  4) Batch list file (.txt, .list, .urls)
       ./transcribe.sh ./inputs.txt ./transcripts/

Output behaviour — single input:
  [output] ends with .txt  -> exact output file
  [output] is a directory  -> file is auto-named inside it
  [output] omitted         -> auto-named file in current directory

Output behaviour — batch list:
  [output] is always treated as a directory.
  [output] omitted -> current directory.
  All items are attempted; failures are collected and reported at the end.

Batch list format:
  - one URL or file path per line
  - blank lines and lines starting with # are ignored
  - relative paths are resolved against the list-file location

Examples:
  ./transcribe.sh "https://youtu.be/dQw4w9WgXcQ" ./youtube_en.txt --language en
  ./transcribe.sh ./recording.wav ./out.txt --language pl --no-timestamps
  ./transcribe.sh ./meeting.mp4 ./transcripts/ --device cuda --compute-type float16
  ./transcribe.sh ./inputs.txt ./transcripts/ --language pl
  ./transcribe.sh "https://example.com/v" ./out.txt --cookies ./cookies.txt
  ./transcribe.sh ./archive.mp3 ./out.txt --model large-v3 --build-mode rebuild
""",
    )

    parser.add_argument(
        "input",
        help="URL, local audio/video file, or batch list (.txt/.list/.urls)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="",
        help="Output .txt file or directory (default: current directory)",
    )

    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        metavar="NAME",
        help=f"Whisper model (default: {_DEFAULT_MODEL}). Examples: small, medium, large-v3",
    )
    parser.add_argument(
        "--language",
        default="",
        metavar="CODE",
        help="ISO 639-1 language code, e.g. pl, en, de. Default: autodetect",
    )
    parser.add_argument(
        "--device",
        default=_DEFAULT_DEVICE,
        choices=sorted(_VALID_DEVICES),
        help=f"Inference device (default: {_DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--compute-type",
        default=_DEFAULT_COMPUTE_TYPE,
        dest="compute_type",
        metavar="TYPE",
        help=(
            f"faster-whisper compute type (default: {_DEFAULT_COMPUTE_TYPE}). "
            "auto selects float16 for cuda, int8 for cpu. "
            f"Explicit choices: {', '.join(sorted(_VALID_COMPUTE_TYPES - {'auto'}))}"
        ),
    )
    parser.add_argument(
        "--tmp-size",
        default=_DEFAULT_TMP_SIZE,
        dest="tmp_size",
        metavar="SIZE",
        help=f"Docker tmpfs size for /tmp (default: {_DEFAULT_TMP_SIZE}). Examples: 8g, 512m",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        dest="no_timestamps",
        help="Write plain transcript without [HH:MM:SS - HH:MM:SS] markers",
    )
    parser.add_argument(
        "--cookies",
        default="",
        metavar="PATH",
        help="Path to yt-dlp cookies.txt for restricted videos",
    )
    parser.add_argument(
        "--image-prefix",
        default=_DEFAULT_IMAGE_PREFIX,
        dest="image_prefix",
        metavar="NAME",
        help=(
            f"Docker image name prefix (default: {_DEFAULT_IMAGE_PREFIX}). "
            "The model is appended as a tag: <prefix>:<model>"
        ),
    )
    parser.add_argument(
        "--build-mode",
        default=_DEFAULT_BUILD_MODE,
        dest="build_mode",
        choices=sorted(_VALID_BUILD_MODES),
        help=(
            f"Image build policy (default: {_DEFAULT_BUILD_MODE}): "
            "auto=build only if missing, rebuild=always rebuild, "
            "fresh=remove image then rebuild without cache"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations and value formats; call _die() on error."""
    if args.compute_type not in _VALID_COMPUTE_TYPES:
        _die(
            f"--compute-type '{args.compute_type}' is not valid. "
            f"Allowed values: {', '.join(sorted(_VALID_COMPUTE_TYPES))}"
        )

    if args.device == "cpu" and args.compute_type == "float16":
        _die("--compute-type float16 is not valid for CPU. Use int8 or float32.")

    if args.device == "cuda" and args.compute_type == "int8":
        _warn(
            "--device cuda with --compute-type int8 works, "
            "but float16 or int8_float16 is typically faster."
        )

    if not _MODEL_RE.match(args.model):
        _die(
            f"--model '{args.model}' contains invalid characters. "
            "Use alphanumeric characters, dots, hyphens or underscores."
        )

    if not _IMAGE_PREFIX_RE.match(args.image_prefix):
        _die(
            f"--image-prefix '{args.image_prefix}' is not a valid Docker image name prefix. "
            "Use alphanumeric characters, dots, colons, slashes, hyphens or underscores."
        )

    if not _TMP_SIZE_RE.match(args.tmp_size):
        _die(
            f"--tmp-size '{args.tmp_size}' is not a valid size. "
            "Provide a number optionally followed by a unit, e.g. 8g, 512m, 2048k."
        )

    if args.language and not _LANGUAGE_RE.match(args.language):
        _die(
            f"--language '{args.language}' does not look like a valid language code. "
            "Use ISO 639-1 codes such as: en, pl, de, zh-CN, pt-BR."
        )

    if args.cookies:
        _validate_cookies_file(args.cookies)


def _validate_cookies_file(path: str) -> None:
    """Check that cookies file is a readable regular file; call _die() on failure."""
    p = Path(path)
    if not p.exists():
        _die(f"Cookies file does not exist: {p}")
    if not p.is_file():
        _die(f"Cookies path is not a regular file: {p}")
    if not os.access(p, os.R_OK):
        _die(f"Cookies file is not readable: {p}")


# ---------------------------------------------------------------------------
# Compute-type resolution
# ---------------------------------------------------------------------------


def _resolve_compute_type(device: str, compute_type: str) -> str:
    """Resolve the ``auto`` compute type to a concrete value based on *device*."""
    if compute_type == "auto":
        return "float16" if device == "cuda" else "int8"
    return compute_type


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------


def _check_docker() -> None:
    """Verify Docker is installed and available; _die() otherwise."""
    try:
        subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except FileNotFoundError:
        _die("Docker is not installed or not available in PATH.")
    except subprocess.CalledProcessError:
        _die(
            "Docker is installed but 'docker --version' returned a non-zero exit code. "
            "Is the Docker daemon running?"
        )


def _image_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _image_has_preloaded_model(image: str, model: str) -> bool:
    result = subprocess.run(
        [
            "docker", "run", "--rm", "--entrypoint", "cat",
            image, "/opt/faster-whisper-cache/preloaded-models.txt",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return model in result.stdout.splitlines()


def _build_image(image: str, model: str, build_mode: str) -> None:
    """Build or verify the Docker image according to *build_mode*."""
    if build_mode == "fresh":
        if _image_exists(image):
            print(f"Removing Docker image: {image}")
            subprocess.run(["docker", "image", "rm", "-f", image], check=True)
        print(f"Building Docker image from scratch (no cache): {image}")
        subprocess.run(
            [
                "docker", "build",
                "--no-cache", "--pull",
                "--build-arg", f"PRELOAD_MODELS={model}",
                "-t", image,
                str(_REPO_ROOT),
            ],
            check=True,
        )
        return

    if build_mode == "rebuild" or not _image_exists(image):
        print(f"Building Docker image: {image}")
        subprocess.run(
            [
                "docker", "build",
                "--build-arg", f"PRELOAD_MODELS={model}",
                "-t", image,
                str(_REPO_ROOT),
            ],
            check=True,
        )
        return

    print(f"Checking preloaded model cache in image for model: {model}")
    if not _image_has_preloaded_model(image, model):
        print(f"Selected model '{model}' is missing in local image cache. Rebuilding image...")
        subprocess.run(
            [
                "docker", "build",
                "--build-arg", f"PRELOAD_MODELS={model}",
                "-t", image,
                str(_REPO_ROOT),
            ],
            check=True,
        )


# ---------------------------------------------------------------------------
# Docker run argument builders
# ---------------------------------------------------------------------------


def _docker_base_args(device: str, tmp_size: str) -> list[str]:
    """Build the ``docker run`` flags that precede the image name."""
    args: list[str] = [
        "run", "--rm",
        "--tmpfs", f"/tmp:rw,size={tmp_size},mode=1777",
        "-e", "HOME=/tmp",
        "-e", "XDG_CACHE_HOME=/tmp/.cache",
        "-e", "HF_HOME=/tmp/.cache/huggingface",
        "-e", "HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub",
        "-e", "TRANSFORMERS_CACHE=/tmp/.cache/huggingface/transformers",
    ]

    for env_var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(env_var)
        if value:
            args += ["-e", f"{env_var}={value}"]

    if device == "cuda":
        args += ["--gpus", "all"]

    try:
        args += ["--user", f"{os.getuid()}:{os.getgid()}"]
    except AttributeError:
        pass  # Windows: os.getuid() / os.getgid() not available

    return args


def _container_base_args(
    model: str,
    device: str,
    compute_type: str,
    language: str,
    no_timestamps: bool,
) -> list[str]:
    """Build the positional arguments passed to the container entrypoint."""
    args: list[str] = [
        "--model", model,
        "--device", device,
        "--compute-type", compute_type,
    ]
    if language:
        args += ["--language", language]
    if no_timestamps:
        args.append("--no-timestamps")
    return args


# ---------------------------------------------------------------------------
# Output resolution
# ---------------------------------------------------------------------------


def _resolve_output(
    output: str,
    force_dir: bool,
) -> tuple[Path, Path | None]:
    """Map the user-supplied *output* string to ``(host_mount_dir, container_output_file)``.

    *container_output_file* is ``None`` when output naming is delegated to the
    container.  When not ``None`` it is the absolute path **inside** the
    container (e.g. ``Path("/out/result.txt")``).
    """
    if not output:
        host_dir = Path.cwd()
        _mkdir_or_raise(host_dir)
        return host_dir, None

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    if force_dir:
        if _is_txt_file(output_path) and not output_path.is_dir():
            raise RuntimeError(
                f"For batch input, [output] must be a directory, "
                f"not a .txt file: {output}"
            )
        _mkdir_or_raise(output_path)
        return output_path, None

    if output_path.is_dir() or output.endswith("/"):
        _mkdir_or_raise(output_path)
        return output_path, None

    if _is_txt_file(output_path):
        _mkdir_or_raise(output_path.parent)
        return output_path.parent, Path("/out") / output_path.name

    # Non-.txt, non-existing, no trailing slash → treat as directory
    _mkdir_or_raise(output_path)
    return output_path, None


# ---------------------------------------------------------------------------
# Single-input runner
# ---------------------------------------------------------------------------


def _run_single(
    input_val: str,
    output: str,
    force_output_dir: bool,
    args: argparse.Namespace,
    effective_image: str,
    compute_type: str,
) -> None:
    """Build and execute the ``docker run`` command for a single input.

    :raises RuntimeError: On any failure (missing file, Docker error, etc.).
    """
    docker_args = _docker_base_args(args.device, args.tmp_size)
    container_args = _container_base_args(
        args.model, args.device, compute_type, args.language, args.no_timestamps
    )

    # Resolve and mount output directory
    host_out_dir, container_out_file = _resolve_output(output, force_output_dir)
    docker_args += ["-v", f"{host_out_dir}:/out"]
    container_args += ["--output-dir", "/out"]
    if container_out_file is not None:
        container_args += ["--output-file", str(container_out_file)]

    # Cookies: file-level bind mount (only the single file, not the whole directory)
    if args.cookies:
        cookies_abs = Path(args.cookies).resolve()
        container_cookies = f"/cookies/{cookies_abs.name}"
        docker_args += ["-v", f"{cookies_abs}:{container_cookies}:ro"]
        container_args += ["--cookies", container_cookies]

    # Input: URL or local file
    if _is_url(input_val):
        container_args += ["--url", input_val]
    else:
        input_path = Path(input_val)
        if not input_path.exists():
            raise RuntimeError(f"Input file does not exist: {input_path}")
        if not input_path.is_file():
            raise RuntimeError(f"Input path is not a regular file: {input_path}")
        if not os.access(input_path, os.R_OK):
            raise RuntimeError(f"Input file is not readable: {input_path}")
        input_abs = input_path.resolve()
        # File-level bind mount: only the specific file is visible inside the container
        container_input = f"/input/{input_abs.name}"
        docker_args += ["-v", f"{input_abs}:{container_input}:ro"]
        container_args += ["--input-file", container_input]

    print(f"Running transcription for: {input_val}")
    cmd = ["docker"] + docker_args + [effective_image] + container_args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"Transcription container exited with non-zero status for: {input_val}"
        )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def _run_batch_file(
    list_file_str: str,
    output: str,
    args: argparse.Namespace,
    effective_image: str,
    compute_type: str,
) -> None:
    """Process each item in a batch list file, collecting errors instead of aborting."""
    list_path = Path(list_file_str)
    if not list_path.exists():
        _die(f"Batch list file does not exist: {list_path}")

    list_abs = list_path.resolve()
    list_dir = list_abs.parent

    items: list[str] = []
    with list_abs.open("r", encoding="utf-8") as fh:
        for line in fh:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            if _is_url(item) or Path(item).is_absolute():
                items.append(item)
            else:
                items.append(str(list_dir / item))

    if not items:
        _warn("Batch list is empty — no actionable items found. Nothing to process.")
        return

    print(f"Batch mode: {len(items)} item(s) to process.")
    errors: list[tuple[str, str]] = []

    for idx, item in enumerate(items, 1):
        print(f"\n[{idx}/{len(items)}] Processing: {item}")
        try:
            _run_single(item, output, True, args, effective_image, compute_type)
        except RuntimeError as exc:
            _warn(f"Failed: {exc}")
            errors.append((item, str(exc)))

    if errors:
        print(f"\n{len(errors)} item(s) failed:", file=sys.stderr)
        for item, err in errors:
            print(f"  {item}: {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nBatch complete: all {len(items)} item(s) processed successfully.")


# ---------------------------------------------------------------------------
# Configuration summary
# ---------------------------------------------------------------------------


def _print_config(
    args: argparse.Namespace,
    compute_type: str,
    effective_image: str,
) -> None:
    print("Selected options:")
    print(f"  input:         {args.input}")
    print(f"  output:        {args.output or '<current-directory>'}")
    print(f"  model:         {args.model}")
    print(f"  language:      {args.language or '<autodetect>'}")
    print(f"  device:        {args.device}")
    print(f"  compute_type:  {compute_type}")
    print(f"  no_timestamps: {args.no_timestamps}")
    print(f"  tmp_size:      {args.tmp_size}")
    print(f"  image:         {effective_image}")
    print(f"  build_mode:    {args.build_mode}")
    print(f"  cookies:       {args.cookies or '<none>'}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _main() -> None:
    args = _parse_args()
    _validate_args(args)

    if not args.input.strip():
        _die("Input must not be empty.")

    compute_type = _resolve_compute_type(args.device, args.compute_type)
    effective_image = f"{args.image_prefix}:{args.model}"

    _print_config(args, compute_type, effective_image)
    _check_docker()

    try:
        _build_image(effective_image, args.model, args.build_mode)
    except subprocess.CalledProcessError as exc:
        _die(f"Docker image build failed (exit code {exc.returncode}).")

    input_val = args.input
    output = args.output

    if (
        not _is_url(input_val)
        and Path(input_val).is_file()
        and _is_list_file(Path(input_val))
    ):
        _run_batch_file(input_val, output, args, effective_image, compute_type)
    else:
        try:
            _run_single(input_val, output, False, args, effective_image, compute_type)
        except RuntimeError as exc:
            _die(str(exc))


def main() -> None:
    try:
        _main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
