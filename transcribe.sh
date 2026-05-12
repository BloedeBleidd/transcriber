#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

if (( BASH_VERSINFO[0] < 4 )); then
  echo "ERROR: Bash 4+ is required. On macOS install bash via Homebrew and run with that shell." >&2
  exit 1
fi

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_PREFIX="transcriber"
MODEL="medium"
LANGUAGE=""
DEVICE="cpu"
COMPUTE_TYPE="auto"
TMP_SIZE="8g"
NO_TIMESTAMPS="0"
BUILD_MODE="auto"
COOKIES_PATH=""
COMPUTE_TYPE_SET_BY_USER="0"

# Portable realpath fallback
get_realpath() {
  if command -v realpath >/dev/null 2>&1; then
    realpath "$1"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c "import os, sys; print(os.path.realpath(sys.argv[1]))" "$1"
  else
    local dir
    local base
    dir="$(cd "$(dirname "$1")" && pwd -P)"
    base="$(basename "$1")"
    printf '%s/%s\n' "$dir" "$base"
  fi
}

usage() {
  cat <<'HELP'
Universal local transcription runner (Docker-based)

Usage:
  ./transcribe.sh <input> [output] [options]

Inputs:
  1) URL supported by yt-dlp
     ./transcribe.sh "https://example.com/video-or-audio-url" ./transcript.txt

  2) Local audio file
     ./transcribe.sh ./audio.mp3 ./transcript.txt

  3) Local video file
     ./transcribe.sh ./video.mp4 ./transcript.txt

  4) Batch list file (.txt, .list, .urls)
     ./transcribe.sh ./inputs.txt ./transcripts

Batch list format:
  - one item per line
  - empty lines are ignored
  - lines starting with # are comments
  - relative local paths are resolved against the list file directory

Output behavior:
  Single input:
    - [output] ends with .txt  => exact output file
    - [output] is directory     => output file is auto-named in that directory
    - [output] omitted          => auto-named file in current directory

  Batch list input:
    - [output] is always treated as output directory
    - [output] omitted => current directory

Options:
  --model <name>            Whisper model (default: medium)
                            examples: small, medium, large-v3

  --language <code>         language code (e.g. pl, en)
                            default: autodetect

  --no-timestamps           write plain transcript (no [HH:MM:SS - HH:MM:SS])

  --cookies <path>          path to yt-dlp cookies.txt for restricted videos

  --device <cpu|cuda>       inference device (default: cpu)
                            Note: 'cuda' requires NVIDIA Container Toolkit and
                            a compatible GPU-enabled image.

  --compute-type <type>     faster-whisper compute type (default: auto)
                            auto picks float16 for cuda, int8 for cpu
                            common values:
                              cpu: int8, float32
                              cuda: float16, int8_float16, int8

  --tmp-size <size>         Docker tmpfs size for /tmp (default: 8g)

  --image-prefix <name>     Docker image name prefix (default: transcriber)
                            The model is appended as a tag: <prefix>:<model>

  --build-mode <mode>       image build policy (default: auto)
                            auto    = build only if image is missing
                            rebuild = always rebuild (uses docker cache)
                            fresh   = remove image + --no-cache --pull

  -h, --help                show this help

Examples:
  ./transcribe.sh "https://youtu.be/dQw4w9WgXcQ" ./youtube_en.txt --language en
  ./transcribe.sh "https://example.com/podcast-episode" ./podcast_es.txt --language es
  ./transcribe.sh ./recording_pl.wav ./recording_pl.txt --language pl --no-timestamps
  ./transcribe.sh ./meeting_en.mp4 ./meeting_en.txt --language en --device cpu
  ./transcribe.sh ./conferencia_es.mp4 ./transcripts --language es --device cuda --compute-type float16
  ./transcribe.sh ./inputs.txt ./transcripts --language pl
  ./transcribe.sh "https://example.com/restricted-video" ./restricted.txt --cookies ./cookies.txt
  ./transcribe.sh ./archive.mp3 ./archive.txt --model large-v3 --language en --build-mode rebuild
  ./transcribe.sh ./demo.mp4 ./demo.txt --build-mode fresh

HELP
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

warn() {
  echo "WARN: $*" >&2
}

require_value() {
  local flag="$1"
  local value="${2:-}"

  if [[ -z "$value" || "$value" == --* ]]; then
    die "$flag requires a value"
  fi
}

is_url() {
  [[ "$1" =~ ^https?:// ]]
}

lowercase() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

is_list_file() {
  local value
  value="$(lowercase "$1")"

  case "$value" in
    *.txt|*.list|*.urls) return 0 ;;
    *) return 1 ;;
  esac
}

is_txt_file() {
  local value
  value="$(lowercase "$1")"

  case "$value" in
    *.txt) return 0 ;;
    *) return 1 ;;
  esac
}

trim_line() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s\n' "$value"
}

abs_existing_file() {
  local path="$1"
  [[ -f "$path" ]] || die "File does not exist: $path"
  [[ -r "$path" ]] || die "File is not readable: $path"
  get_realpath "$path"
}

abs_output_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$PWD" "$path"
  fi
}

validate_runtime_options() {
  case "$DEVICE" in
    cpu|cuda) ;;
    *) die "--device must be cpu or cuda" ;;
  esac

  case "$COMPUTE_TYPE" in
    auto|int8|float32|float16|int8_float16) ;;
    *)
      die "--compute-type must be auto, int8, float32, float16 or int8_float16"
      ;;
  esac

  if [[ "$DEVICE" == "cuda" && "$COMPUTE_TYPE" == "int8" ]]; then
    warn "--device cuda with --compute-type int8 works, but float16 or int8_float16 is typically faster."
  fi

  if [[ "$DEVICE" == "cpu" && "$COMPUTE_TYPE" == "float16" ]]; then
    if [[ "$COMPUTE_TYPE_SET_BY_USER" == "1" ]]; then
      die "--compute-type float16 is not valid for CPU"
    fi
    COMPUTE_TYPE="int8"
    warn "Adjusted compute-type to int8 because CPU does not support float16."
  fi
}

resolve_runtime_defaults() {
  if [[ "$COMPUTE_TYPE" == "auto" ]]; then
    if [[ "$DEVICE" == "cuda" ]]; then
      COMPUTE_TYPE="float16"
    else
      COMPUTE_TYPE="int8"
    fi
  fi
}

print_effective_configuration() {
  local input="$1"
  local output="$2"
  local effective_image="${IMAGE_PREFIX}:${MODEL}"

  echo "Selected options:"
  echo "  input: ${input}"
  echo "  output: ${output:-<current-directory>}"
  echo "  model: ${MODEL}"
  echo "  language: ${LANGUAGE:-<autodetect>}"
  echo "  device: ${DEVICE}"
  echo "  compute_type: ${COMPUTE_TYPE}"
  echo "  no_timestamps: ${NO_TIMESTAMPS}"
  echo "  tmp_size: ${TMP_SIZE}"
  echo "  image: ${effective_image}"
  echo "  build_mode: ${BUILD_MODE}"
  echo "  cookies: ${COOKIES_PATH:-<none>}"
}

build_image_if_needed() {
  command -v docker >/dev/null 2>&1 || die "Docker is not installed or not available in PATH"

  local effective_image="${IMAGE_PREFIX}:${MODEL}"

  if [[ "$BUILD_MODE" == "fresh" ]]; then
    if docker image inspect "$effective_image" >/dev/null 2>&1; then
      echo "Removing Docker image: $effective_image"
      docker image rm -f "$effective_image"
    fi

    echo "Building Docker image from scratch (no cache): $effective_image"
    docker build --no-cache --pull --build-arg "PRELOAD_MODELS=${MODEL}" -t "$effective_image" "$SCRIPT_DIR"
    return 0
  fi

  if [[ "$BUILD_MODE" == "rebuild" ]] || ! docker image inspect "$effective_image" >/dev/null 2>&1; then
    echo "Building Docker image: $effective_image"
    docker build --build-arg "PRELOAD_MODELS=${MODEL}" -t "$effective_image" "$SCRIPT_DIR"
    return 0
  fi

  echo "Checking preloaded model cache in image for model: $MODEL"
  if ! image_has_preloaded_model "$MODEL"; then
    echo "Selected model '$MODEL' is missing in local image cache. Rebuilding image..."
    docker build --build-arg "PRELOAD_MODELS=${MODEL}" -t "$effective_image" "$SCRIPT_DIR"
  fi
}

image_has_preloaded_model() {
  local model_name="$1"
  local effective_image="${IMAGE_PREFIX}:${MODEL}"

  docker run --rm \
    --entrypoint cat \
    "$effective_image" /opt/faster-whisper-cache/preloaded-models.txt 2>/dev/null \
    | grep -Fxq "$model_name"
}

add_common_docker_args() {
  local -n docker_args_ref="$1"

  docker_args_ref=(
    run
    --rm
    --tmpfs "/tmp:rw,size=${TMP_SIZE},mode=1777"
    -e "HOME=/tmp"
    -e "XDG_CACHE_HOME=/tmp/.cache"
    -e "HF_HOME=/tmp/.cache/huggingface"
    -e "HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub"
    -e "TRANSFORMERS_CACHE=/tmp/.cache/huggingface/transformers"
  )

  if [[ -n "${HF_TOKEN:-}" ]]; then
    docker_args_ref+=(-e "HF_TOKEN=${HF_TOKEN}")
  fi

  if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    docker_args_ref+=(-e "HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}")
  fi

  if [[ "$DEVICE" == "cuda" ]]; then
    docker_args_ref+=(--gpus all)
  fi

  if command -v id >/dev/null 2>&1; then
    docker_args_ref+=(--user "$(id -u):$(id -g)")
  fi
}

add_common_container_args() {
  local -n container_args_ref="$1"

  container_args_ref+=(--model "$MODEL")
  container_args_ref+=(--device "$DEVICE")
  container_args_ref+=(--compute-type "$COMPUTE_TYPE")

  if [[ -n "$LANGUAGE" ]]; then
    container_args_ref+=(--language "$LANGUAGE")
  fi

  if [[ "$NO_TIMESTAMPS" == "1" ]]; then
    container_args_ref+=(--no-timestamps)
  fi
}

add_cookies_args() {
  local -n docker_args_ref="$1"
  local -n container_args_ref="$2"

  if [[ -z "$COOKIES_PATH" ]]; then
    return 0
  fi

  local cookies_abs
  local cookies_dir
  local cookies_file

  cookies_abs="$(abs_existing_file "$COOKIES_PATH")"
  cookies_dir="$(dirname "$cookies_abs")"
  cookies_file="$(basename "$cookies_abs")"

  docker_args_ref+=(-v "${cookies_dir}:/cookies:ro")
  container_args_ref+=(--cookies "/cookies/${cookies_file}")
}

add_output_args() {
  local output="$1"
  local force_dir="$2"
  local -n docker_args_ref="$3"
  local -n container_args_ref="$4"

  local output_abs
  local output_dir
  local output_file

  if [[ -z "$output" ]]; then
    output_abs="$PWD"
    mkdir -p "$output_abs"
    docker_args_ref+=(-v "${output_abs}:/out")
    container_args_ref+=(--output-dir /out)
    return 0
  fi

  output_abs="$(abs_output_path "$output")"

  if [[ "$force_dir" == "1" ]]; then
    if is_txt_file "$output"; then
      die "For batch input, output must be a directory, not a .txt file: $output"
    fi

    mkdir -p "$output_abs"
    docker_args_ref+=(-v "${output_abs}:/out")
    container_args_ref+=(--output-dir /out)
    return 0
  fi

  if [[ -d "$output_abs" || "$output" == */ ]]; then
    mkdir -p "$output_abs"
    docker_args_ref+=(-v "${output_abs}:/out")
    container_args_ref+=(--output-dir /out)
    return 0
  fi

  if is_txt_file "$output_abs"; then
    output_dir="$(dirname "$output_abs")"
    output_file="$(basename "$output_abs")"

    mkdir -p "$output_dir"
    docker_args_ref+=(-v "${output_dir}:/out")
    container_args_ref+=(--output-dir /out --output-file "/out/${output_file}")
    return 0
  fi

  mkdir -p "$output_abs"
  docker_args_ref+=(-v "${output_abs}:/out")
  container_args_ref+=(--output-dir /out)
}

run_single() {
  local input="$1"
  local output="$2"
  local force_output_dir="${3:-0}"

  local docker_args=()
  local container_args=()

  add_common_docker_args docker_args
  add_common_container_args container_args
  add_cookies_args docker_args container_args
  add_output_args "$output" "$force_output_dir" docker_args container_args

  local effective_image="${IMAGE_PREFIX}:${MODEL}"

  if is_url "$input"; then
    container_args+=(--url "$input")
  else
    local input_abs
    local input_dir
    local input_file

    input_abs="$(abs_existing_file "$input")"
    input_dir="$(dirname "$input_abs")"
    input_file="$(basename "$input_abs")"

    docker_args+=(-v "${input_dir}:/input:ro")
    container_args+=(--input-file "/input/${input_file}")
  fi

  echo "Running transcription for: $input"
  docker "${docker_args[@]}" "$effective_image" "${container_args[@]}"
}

run_batch_file() {
  local list_file="$1"
  local output="$2"

  local list_abs
  local list_dir
  local line
  local item
  local resolved_item

  list_abs="$(abs_existing_file "$list_file")"
  list_dir="$(dirname "$list_abs")"

  while IFS= read -r line || [[ -n "$line" ]]; do
    item="$(trim_line "$line")"

    if [[ -z "$item" || "$item" == \#* ]]; then
      continue
    fi

    if is_url "$item"; then
      resolved_item="$item"
    elif [[ "$item" = /* ]]; then
      resolved_item="$item"
    else
      resolved_item="${list_dir}/${item}"
    fi

    run_single "$resolved_item" "$output" "1"
  done < "$list_abs"
}

main() {
  local input=""
  local output=""
  local positional=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        require_value "$1" "${2:-}"
        MODEL="$2"
        shift 2
        ;;
      --language)
        require_value "$1" "${2:-}"
        LANGUAGE="$2"
        shift 2
        ;;
      --device)
        require_value "$1" "${2:-}"
        DEVICE="$2"
        DEVICE_SET_BY_USER="1"
        shift 2
        ;;
      --compute-type)
        require_value "$1" "${2:-}"
        COMPUTE_TYPE="$2"
        COMPUTE_TYPE_SET_BY_USER="1"
        shift 2
        ;;
      --tmp-size)
        require_value "$1" "${2:-}"
        TMP_SIZE="$2"
        shift 2
        ;;
      --cookies)
        require_value "$1" "${2:-}"
        COOKIES_PATH="$2"
        shift 2
        ;;
      --image-prefix)
        require_value "$1" "${2:-}"
        IMAGE_PREFIX="$2"
        shift 2
        ;;
      --build-mode)
        require_value "$1" "${2:-}"
        BUILD_MODE="$2"
        shift 2
        ;;
      --no-timestamps)
        NO_TIMESTAMPS="1"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        while [[ $# -gt 0 ]]; do
          positional+=("$1")
          shift
        done
        ;;
      -*)
        die "Unknown option: $1"
        ;;
      *)
        positional+=("$1")
        shift
        ;;
    esac
  done

  if [[ ${#positional[@]} -lt 1 ]]; then
    usage
    exit 1
  fi

  if [[ ${#positional[@]} -gt 2 ]]; then
    die "Expected at most 2 positional arguments: <input> [output]"
  fi

  input="${positional[0]}"
  output="${positional[1]:-}"

  [[ -n "$input" ]] || die "Input must not be empty"

  validate_runtime_options
  resolve_runtime_defaults
  validate_runtime_options
  case "$BUILD_MODE" in
    auto|rebuild|fresh) ;;
    *) die "--build-mode must be one of: auto, rebuild, fresh" ;;
  esac
  print_effective_configuration "$input" "$output"
  build_image_if_needed

  if ! is_url "$input" && [[ -f "$input" ]] && is_list_file "$input"; then
    run_batch_file "$input" "$output"
  else
    run_single "$input" "$output" "0"
  fi
}

main "$@"
