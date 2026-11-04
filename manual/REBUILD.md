# Rebuilding the reference manual

The 56-page source-edition 3.8.0 manual includes a beginner tutorial, checked
examples, a public Python API reference, and a registered-name catalogue.
The source snapshot documented by this edition is
`6d067221be924fe789203ed0fc0a3a8b4a656521`; PyPI versions may differ.

From the repository root, with core dependencies installed:

```bash
python scripts/check_manual_examples.py
PYTHONPATH=. python scripts/beginner_tutorial.py
cd manual
tectonic -X compile mikoshilang-manual.tex
```

The second command uses POSIX shell syntax. On Windows PowerShell set
`$env:PYTHONPATH = '.'` then run `python scripts/beginner_tutorial.py`.

Tectonic downloads required TeX packages on its first run. The output is
`manual/mikoshilang-manual.pdf`.

## Source files

- `mikoshilang-manual.tex`: main document and installation/reference overview.
- `beginner-tutorial.tex`: authored beginner chapter.
- `current-examples.tex`: generated checked-example chapter.
- `generated-reference.tex`: API reference and registered-name catalogue snapshot.

To regenerate the checked-example chapter after intentional example changes:

```bash
python scripts/check_manual_examples.py --write
```

The checker runs ten standalone programs with timeouts, verifies their assertions
and exact expected output, and checks the generated chapter for drift. The beginner
script checks the tutorial calculations. These are bounded documentation checks,
not certification of the entire library. Keep the catalogue aligned with its stated
source snapshot when updating library versions; do not change inventory counts
without regenerating and auditing the associated reference.
