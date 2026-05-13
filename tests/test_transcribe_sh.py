import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "transcribe.sh"


class TranscribeShellTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="transcriber-shell-tests-")
        self.tmp_path = Path(self.temp_dir.name)
        self.fakebin = self.tmp_path / "bin"
        self.fakebin.mkdir()
        self.docker_log = self.tmp_path / "docker.log"
        self.media_file = self.tmp_path / "sample.mp4"
        self.media_file.write_text("sample", encoding="utf-8")
        self.cookies_file = self.tmp_path / "cookies.txt"
        self.cookies_file.write_text("cookies", encoding="utf-8")
        self._write_fake_docker()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_fake_docker(self) -> None:
        docker_path = self.fakebin / "docker"
        docker_path.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                set -euo pipefail
                {{
                  echo "__CALL__"
                  printf '%s\\n' "$@"
                }} >> "{self.docker_log}"
                case "${{1:-}}" in
                  --version)
                    echo "Docker version 26.1.0"
                    ;;
                  image)
                    if [[ "${{2:-}}" == "inspect" ]]; then
                      exit 0
                    fi
                    ;;
                  run)
                    if [[ "${{2:-}}" == "--rm" && "${{3:-}}" == "--entrypoint" && "${{4:-}}" == "cat" ]]; then
                      echo "medium"
                      exit 0
                    fi
                    exit 0
                    ;;
                  build)
                    exit 0
                    ;;
                esac
                exit 0
                """
            ),
            encoding="utf-8",
        )
        docker_path.chmod(0o755)

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PATH"] = f"{self.fakebin}:{env['PATH']}"
        return subprocess.run(
            ["bash", str(SCRIPT_PATH), *args],
            cwd=self.tmp_path,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _docker_calls(self) -> list[list[str]]:
        if not self.docker_log.exists():
            return []

        calls: list[list[str]] = []
        current: list[str] = []
        for line in self.docker_log.read_text(encoding="utf-8").splitlines():
            if line == "__CALL__":
                if current:
                    calls.append(current)
                current = []
                continue
            current.append(line)
        if current:
            calls.append(current)
        return calls

    def test_cpu_int8_float16_is_rejected(self) -> None:
        result = self._run(
            str(self.media_file),
            str(self.tmp_path / "out.txt"),
            "--device",
            "cpu",
            "--compute-type",
            "int8_float16",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "--compute-type int8_float16 is not valid for CPU. Use int8 or float32.",
            result.stderr,
        )
        self.assertEqual(self._docker_calls(), [])

    def test_invalid_build_mode_fails_before_docker(self) -> None:
        result = self._run(
            "https://example.com/media",
            str(self.tmp_path / "out.txt"),
            "--build-mode",
            "broken",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--build-mode must be one of: auto, rebuild, fresh", result.stderr)
        self.assertEqual(self._docker_calls(), [])

    def test_cuda_auto_resolves_to_float16_and_enables_gpu(self) -> None:
        result = self._run(
            "https://example.com/media",
            str(self.tmp_path / "out.txt"),
            "--device",
            "cuda",
            "--compute-type",
            "auto",
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        run_call = self._docker_calls()[-1]
        self.assertIn("--gpus", run_call)
        self.assertIn("all", run_call)
        self.assertIn("--compute-type", run_call)
        self.assertIn("float16", run_call)

    def test_url_runs_with_no_new_privileges(self) -> None:
        result = self._run(
            "https://example.com/media",
            str(self.tmp_path / "out.txt"),
            "--cookies",
            str(self.cookies_file),
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        run_call = self._docker_calls()[-1]
        self.assertIn("--security-opt", run_call)
        self.assertIn("no-new-privileges", run_call)


if __name__ == "__main__":
    unittest.main()
