#!/usr/bin/env python3
"""Run the library's executable Check: lines and independent check tables (see mikoshilang/verification.py).

    python scripts/verify_library.py              # heads with a Check: line + identity/mpmath/NumPy tables (CI)
    python scripts/verify_library.py --all        # every audited head's reference cases as well (slow)
    python scripts/verify_library.py --names Gamma Zeta --markdown report.md

Exit status 1 when any check fails.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mikoshilang.verification import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
