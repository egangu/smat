"""Regressions for CPU-only Colab and dependency installation."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import demo_setup as setup


class RuntimeTest(unittest.TestCase):
    def select(self, build, available, request="auto", failure=None):
        torch = SimpleNamespace(
            version=SimpleNamespace(cuda=build), __version__="test",
            cuda=SimpleNamespace(is_available=Mock(return_value=available)),
            empty=Mock(side_effect=failure),
        )
        output = io.StringIO()
        with patch.dict(sys.modules, {"torch": torch}), patch.dict(
            setup.os.environ, {"SMAT_DEMO_DEVICE": request}
        ), redirect_stdout(output):
            device = setup.select_device()
        return device, output.getvalue(), torch

    def test_cpu_only_torch_with_explicit_cuda_falls_back(self):
        device, output, torch = self.select(None, False, "cuda")
        self.assertEqual(device, "cpu")
        self.assertIn("CPU-only", output)
        self.assertIn("Disconnect and delete runtime", output)
        torch.empty.assert_not_called()

    def test_working_cuda_is_preferred(self):
        device, _, torch = self.select("12.8", True)
        self.assertEqual(device, "cuda")
        torch.empty.assert_called_once_with(1, device="cuda")

    def test_cuda_build_without_gpu_falls_back(self):
        self.assertEqual(self.select("12.8", False)[0], "cpu")

    def test_cuda_initialization_failure_falls_back(self):
        device, output, _ = self.select("12.8", True, failure=RuntimeError("driver"))
        self.assertEqual(device, "cpu")
        self.assertIn("driver", output)

    def test_explicit_cpu_skips_gpu_probe(self):
        device, _, torch = self.select("12.8", True, "cpu")
        self.assertEqual(device, "cpu")
        torch.cuda.is_available.assert_not_called()
        torch.empty.assert_not_called()

    def test_invalid_device_is_rejected(self):
        with self.assertRaises(ValueError):
            self.select("12.8", True, "typo")

    def test_dependency_install_pins_existing_torch(self):
        commands = []
        def pip(command):
            commands.append(command)
            if "--constraint" in command:
                path = Path(command[command.index("--constraint") + 1])
                self.assertEqual(path.read_text(),
                                 "torch==2.9.1+cu128\ntorchvision==0.24.1+cu128\n")
        versions = {"torch": "2.9.1+cu128", "torchvision": "0.24.1+cu128"}
        updates = SimpleNamespace(FTStepper=object, SMATStepper=object)
        with patch.object(setup.importlib.util, "find_spec",
                          side_effect=lambda name: None if name == "timm" else object()), \
             patch.object(setup.importlib.metadata, "version", side_effect=versions.__getitem__), \
             patch.object(setup.subprocess, "check_call", side_effect=pip), \
             patch.dict(sys.modules, {"smat.train.updates": updates}):
            setup.ensure_runtime()
        self.assertEqual(len(commands), 1)
        self.assertIn("timm==1.0.30", commands[0])
        self.assertNotIn("--upgrade", commands[0])


if __name__ == "__main__":
    unittest.main()
