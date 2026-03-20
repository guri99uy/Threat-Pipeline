"""Suppress noisy third-party warnings before any library imports.

Import this module as early as possible — before tensorflow, transformers,
or huggingface_hub are loaded. All suppression is best-effort: if a library
is already imported, its warnings may have already fired.
"""

import os
import warnings
import logging
import threading

# ── Environment variables (must be set before TF/HF import) ──────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")

# ── Python warning filters ───────────────────────────────────────────────
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*The name tf\\..*is deprecated.*")
warnings.filterwarnings("ignore", message=".*gfile.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# ── Logger levels ────────────────────────────────────────────────────────
for _name in (
    "tensorflow",
    "tensorflow_hub",
    "tf_keras",
    "transformers",
    "transformers.modeling_utils",
    "huggingface_hub",
    "huggingface_hub.utils._http",
    "absl",
):
    logging.getLogger(_name).setLevel(logging.ERROR)

# Disable transformers progress bars and load reports
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ── Suppress safetensors background thread crash ─────────────────────────
_original_excepthook = threading.excepthook


def _quiet_thread_excepthook(args):
    if args.exc_type is OSError and "safetensors" in str(args.exc_value):
        return
    _original_excepthook(args)


threading.excepthook = _quiet_thread_excepthook
