"""Run a .mkl MikoshiLang source file in a fresh Session and print the final result.

Usage: python examples/run_mkl.py examples/06_language.mkl
"""
import pathlib
import sys

from mikoshilang import Session

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    result = Session().evaluate(pathlib.Path(sys.argv[1]).read_text())
    if result is not None and str(result) != "Null":
        print(result)
