"""
Environment validation script — do not modify.

Run this after installing requirements.txt to confirm your environment is correct:
    python testenv.py

Expected output:
    pandas: <version>
    numpy.__version__: <version>
    torch.__version__: <version>
    Environment OK
"""

import sys

import pandas


def check_environment():
    try:
        import pandas as pd
        import numpy as np
        import torch
        import torch.nn as nn
        print(f"pandas: {pandas.__version__}")
        print("numpy.__version__ =", np.__version__,)
        print("torch.__version__ =", torch.__version__,)

        print("Environment OK")
    except ImportError as e:
        print(f"ImportError: {e}", file=sys.stderr)
        print(
            "\nYour environment is missing a required package.",
            file=sys.stderr,
        )
        print(
            "Confirm your venv is active ((.venv) prefix in prompt), then run:",
            file=sys.stderr,
        )
        print("    pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    check_environment()