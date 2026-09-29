"""Install only the dependencies needed by the tiny notebook."""
import importlib.util
import importlib.metadata
import platform
import subprocess
import sys

SMAT_REVISION = "4681a8c40998f59d7b0759ee44cebfa791b47409"


def ensure_runtime():
    def install(*requirements):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *requirements])

    if importlib.util.find_spec("torch") is None:
        options = ("--index-url", "https://download.pytorch.org/whl/cpu") if platform.system() != "Darwin" else ()
        install("torch>=2.9", *options)
    requirements = {"numpy": "numpy>=1.24", "matplotlib": "matplotlib>=3.7",
                    "pandas": "pandas>=2", "timm": "timm==1.0.30"}
    missing = [spec for module, spec in requirements.items() if importlib.util.find_spec(module) is None]
    if importlib.util.find_spec("timm") and importlib.metadata.version("timm") != "1.0.30":
        missing.append("timm==1.0.30")
    if missing:
        install(*missing)
    try:
        from smat.train.updates import FTStepper, SMATStepper
    except ImportError:
        install("--no-deps", f"git+https://github.com/egangu/smat.git@{SMAT_REVISION}")
