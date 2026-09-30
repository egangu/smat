"""Prepare the demo without replacing an existing PyTorch installation."""
import importlib.util
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile


SMAT_REVISION = "4681a8c40998f59d7b0759ee44cebfa791b47409"
COLAB_RESET = (
    "In Colab, use Runtime > Disconnect and delete runtime, select a GPU "
    "under Change runtime type, reconnect, then run all cells."
)


def ensure_runtime():
    def install(*requirements, constraints=None):
        command = [sys.executable, "-m", "pip", "install", "--quiet"]
        if constraints:
            command += ["--constraint", constraints]
        subprocess.check_call(command + list(requirements))

    if importlib.util.find_spec("torch") is None:
        if "google.colab" in sys.modules:
            raise RuntimeError("Colab's preinstalled PyTorch is missing. " + COLAB_RESET)
        # Use the standard distribution, never force a CPU-only wheel on a GPU host.
        install("torch>=2.9", "torchvision")

    requirements = {"numpy": "numpy>=1.24", "matplotlib": "matplotlib>=3.7",
                    "pandas": "pandas>=2", "timm": "timm==1.0.30"}
    missing = [spec for module, spec in requirements.items()
               if importlib.util.find_spec(module) is None]
    if importlib.util.find_spec("timm") and importlib.metadata.version("timm") != "1.0.30":
        missing.append("timm==1.0.30")
    if missing:
        # timm depends on torchvision, which can otherwise replace torch.
        pinned = []
        for name in ("torch", "torchvision"):
            try:
                pinned.append(f"{name}=={importlib.metadata.version(name)}")
            except importlib.metadata.PackageNotFoundError:
                pass
        with tempfile.TemporaryDirectory() as directory:
            constraints = Path(directory) / "torch-constraints.txt"
            constraints.write_text("\n".join(pinned) + "\n")
            install(*missing, constraints=str(constraints))
    try:
        from smat.train.updates import FTStepper, SMATStepper
    except ImportError:
        install("--no-deps", f"git+https://github.com/egangu/smat.git@{SMAT_REVISION}")


def select_device():
    """Prefer working CUDA; report the reason whenever we fall back to CPU."""
    import torch

    requested = os.environ.get("SMAT_DEMO_DEVICE", "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("SMAT_DEMO_DEVICE must be auto, cpu or cuda")
    device = "cpu"
    if requested != "cpu":
        reason = None
        if torch.version.cuda is None:
            reason = "This PyTorch installation is CPU-only."
        elif not torch.cuda.is_available():
            reason = "PyTorch cannot access a CUDA GPU."
        else:
            try:
                torch.empty(1, device="cuda")  # Check actual CUDA initialization.
                device = "cuda"
            except (RuntimeError, AssertionError) as error:
                reason = f"CUDA initialization failed: {error}"
        if reason:
            print(f"Using CPU. {reason} {COLAB_RESET}")
    print(f"Device: {device} | PyTorch: {torch.__version__} | CUDA build: {torch.version.cuda}")
    return device
