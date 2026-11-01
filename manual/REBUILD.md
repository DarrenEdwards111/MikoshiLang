# Manual rebuild instructions

The PDF is built from `mikoshilang-manual.tex`. The 3.8.0 PDF was built with
[Tectonic](https://tectonic-typesetting.github.io/) (XeTeX), which fetches the LaTeX packages it needs:

```bash
cd manual
tectonic -X compile mikoshilang-manual.tex
```

`pdflatex` also works if a TeX Live installation is available (run it twice for the table of contents):

```bash
cd manual
pdflatex mikoshilang-manual.tex
pdflatex mikoshilang-manual.tex
```

## 3.8.0 update

- Version 3.7.0 → 3.8.0.
- Replaced the retracted "6,324 functions" figure with the coverage roadmap figure (4,402 of 6,716 documented
  Wolfram symbols), and added a "Status of this manual" note: where the manual conflicts with the README or
  `docs/`, those take precedence.
- Removed the dead `mikoshi.co.uk` console and REST API links; the hosted services are offline.
- Added the verification, Python interop and types pillars to the key features.

Most sections below the introduction still describe pre-3.8 releases and need a full rewrite.
