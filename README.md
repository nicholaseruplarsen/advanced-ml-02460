# Mini-project 1 for 02460 Advanced Machine Learning

## Setup

```bash
uv sync
```

## Run experiments

```bash
# Gaussian prior
uv run python src/vae_bernoulli.py train --prior gaussian --epochs 10 --model vae_gaussian.pt
uv run python src/vae_bernoulli.py evaluate --prior gaussian --model vae_gaussian.pt
uv run python src/vae_bernoulli.py plot_posterior --prior gaussian --model vae_gaussian.pt --posterior-plot posterior_gaussian.png

# MoG prior
uv run python src/vae_bernoulli.py train --prior mog --epochs 10 --model vae_mog.pt
uv run python src/vae_bernoulli.py evaluate --prior mog --model vae_mog.pt
uv run python src/vae_bernoulli.py plot_posterior --prior mog --model vae_mog.pt --posterior-plot posterior_mog.png

# Flow prior
uv run python src/vae_bernoulli.py train --prior flow --epochs 10 --model vae_flow.pt
uv run python src/vae_bernoulli.py evaluate --prior flow --model vae_flow.pt
uv run python src/vae_bernoulli.py plot_posterior --prior flow --model vae_flow.pt --posterior-plot posterior_flow.png
```

### Extra:

```bash
# Sample from a trained VAE checkpoint
uv run python src/vae_bernoulli.py sample --prior flow --model vae_flow.pt --samples samples_flow.png

# Use Gaussian decoder / continuous MNIST variant
uv run python src/vae_bernoulli.py train --prior gaussian --decoder gaussian --continuous --fixed-var --model vae_gauss_decoder.pt
```
## Report

Source: `report/template.tex` (bibliography: `report/template.bib`)

Build from the `report/` directory:

```bash
cd report && latexmk -pdf template.tex
```
