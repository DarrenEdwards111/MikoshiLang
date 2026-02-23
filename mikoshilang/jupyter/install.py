"""Install the MikoshiLang Jupyter kernel."""

import json
import os
import sys
from pathlib import Path


def install_kernel(user=True, prefix=None):
    """Install the MikoshiLang kernel spec."""
    from jupyter_client.kernelspec import KernelSpecManager

    kernel_json = {
        "argv": [sys.executable, "-m", "mikoshilang.jupyter", "-f", "{connection_file}"],
        "display_name": "MikoshiLang",
        "language": "mikoshilang",
    }

    # Write to a temp dir then install
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        kernel_dir = Path(td) / "mikoshilang"
        kernel_dir.mkdir()
        with open(kernel_dir / "kernel.json", "w") as f:
            json.dump(kernel_json, f, indent=2)

        ksm = KernelSpecManager()
        ksm.install_kernel_spec(str(kernel_dir), kernel_name="mikoshilang", user=user, prefix=prefix)

    print("MikoshiLang kernel installed successfully!")
    print("Launch with: jupyter notebook (then select MikoshiLang kernel)")


def main():
    install_kernel()


if __name__ == "__main__":
    main()
