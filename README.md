# Mini-project 1 for 02460 Advanced Machine Learning.

## Setup

```bash
uv sync
```

## Running code

```bash
uv run python -c "from src.fid import compute_fid; ..."
# or
uv run src/some_script.py
```

## Report

Source: `tex/template.tex` (bibliography: `tex/template.bib`)

Build from the `tex/` directory:

```bash
cd tex && pdflatex template.tex && bibtex template && pdflatex template.tex && pdflatex template.tex
```
