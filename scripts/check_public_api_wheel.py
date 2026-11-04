#!/usr/bin/env python
"""Run the public-API suites against an installed wheel, outside the checkout.

Install the wheel into this interpreter first. Optional scientific/property
suites skip when their dependencies are absent; core tests must still collect.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True, type=Path,
                        help='Parent directory for the isolated temporary checkout of tests')
    parser.add_argument('--junitxml', type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(set((repository / 'tests').glob('test_public_api_*.py')) |
                   set((repository / 'tests').glob('test_python_*_regressions.py')))
    if not files:
        parser.error('No public-API test files found')
    manifest = repository / 'tests' / 'public_api_audit.json'
    if not manifest.is_file():
        parser.error('Public-API evidence manifest is missing')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['MPLBACKEND'] = 'Agg'
    with tempfile.TemporaryDirectory(prefix='mikoshi-public-api-', dir=str(args.work_dir)) as temporary:
        isolated = Path(temporary)
        tests = isolated / 'tests'
        tests.mkdir()
        for source in [*files, manifest]:
            shutil.copy2(source, tests / source.name)
        smoke = (
            'from pathlib import Path; import mikoshilang; '
            'p=Path(mikoshilang.__file__).resolve(); '
            'assert "site-packages" in p.parts, str(p); '
            'print("Installed wheel:", p)'
        )
        subprocess.run([sys.executable, '-c', smoke], cwd=isolated, env=env, check=True)
        command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests', '--tb=short']
        if args.junitxml:
            command.append('--junitxml=' + str(args.junitxml.resolve()))
        return subprocess.run(command, cwd=isolated, env=env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
