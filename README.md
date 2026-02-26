# Mini-project 1 for 02460 Advanced Machine Learning.

## Setup

```bash
uv sync
```

## Running code

```bash
uv run python src/vae_bernoulli.py train --prior flow --epochs 50
uv run python src/flow.py train --data mnist --mask half --num-transformations 8
uv run python src/vae_bernoulli.py evaluate --prior flow --model model.pt

# or import
# from src.fid import compute_fid
# from src.flow import FlowPrior
```

## Report

Source: `tex/template.tex` (bibliography: `tex/template.bib`)

Build from the `tex/` directory:

```bash
cd tex && latexmk -pdf template.tex
```
