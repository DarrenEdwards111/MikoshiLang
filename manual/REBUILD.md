# Manual Rebuild Instructions

The manual has been updated with current function counts and domains.

## To regenerate the PDF:

```bash
cd manual
pdflatex mikoshilang-manual.tex
pdflatex mikoshilang-manual.tex  # Run twice for TOC
```

## Or using Docker:

```bash
docker run --rm -v $(pwd):/work texlive/texlive pdflatex mikoshilang-manual.tex
```

## Changes in this update:

- Version: 2.0.0 → 3.1.1
- Function count: 1,794 → 6,327
- Domains: 25+ → 40+
- Added comprehensive list of all 32 extended modules
- Updated domain coverage section with new fields
