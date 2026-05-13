import argparse
import importlib.util
import sys
import types
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "app" / "transcriber.py"


def load_transcriber_module():
    yt_dlp_module = types.ModuleType("yt_dlp")
    yt_dlp_module.DownloadError = RuntimeError
    yt_dlp_module.YoutubeDL = object

    faster_whisper_module = types.ModuleType("faster_whisper")

    class WhisperModel:
        pass

    faster_whisper_module.WhisperModel = WhisperModel

    tqdm_module = types.ModuleType("tqdm")

    class DummyTqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total")
            self.n = 0

        def update(self, amount):
            self.n += amount

        def close(self):
            return None

    tqdm_module.tqdm = DummyTqdm

    previous_modules = {
        name: sys.modules.get(name)
        for name in ("yt_dlp", "faster_whisper", "tqdm")
    }
    sys.modules["yt_dlp"] = yt_dlp_module
    sys.modules["faster_whisper"] = faster_whisper_module
    sys.modules["tqdm"] = tqdm_module

    try:
        spec = importlib.util.spec_from_file_location("transcriber_under_test", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in previous_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class ValidateArgsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_transcriber_module()

    def test_rejects_cpu_int8_float16(self) -> None:
        args = argparse.Namespace(
            output_file="out.txt",
            compute_type="int8_float16",
            device="cpu",
            language=None,
        )

        with self.assertRaises(SystemExit) as ctx:
            self.module.validate_args(args)

        self.assertEqual(
            str(ctx.exception),
            "--compute-type int8_float16 is not valid for CPU. Use int8 or float32.",
        )

    def test_allows_cuda_int8_float16(self) -> None:
        args = argparse.Namespace(
            output_file="out.txt",
            compute_type="int8_float16",
            device="cuda",
            language="en",
        )

        self.module.validate_args(args)


if __name__ == "__main__":
    unittest.main()
