# Mini-project 1 for 02460 Advanced Machine Learning

## Setup

```bash
uv sync
```

## Part A: VAE Priors (binarized MNIST)

All VAE commands run from the project root. Add `--device mps` (Apple Silicon) or `--device cuda` (NVIDIA) to use GPU.

### 1) VAE with Gaussian prior

```bash
# Train
uv run python src/vae_bernoulli.py train --prior gaussian --epochs 10 --model vae_gaussian.pt

# Evaluate test ELBO
uv run python src/vae_bernoulli.py evaluate --prior gaussian --model vae_gaussian.pt

# Sample
uv run python src/vae_bernoulli.py sample --prior gaussian --model vae_gaussian.pt --samples samples_gaussian.png

# Plot prior vs aggregate posterior
uv run python src/vae_bernoulli.py plot_prior_posterior --prior gaussian --model vae_gaussian.pt --prior-posterior-plot prior_posterior_gaussian.png
```

### 2) VAE with Mixture of Gaussians (MoG) prior

```bash
# Train
uv run python src/vae_bernoulli.py train --prior mog --num-components 10 --epochs 10 --model vae_mog.pt

# Evaluate test ELBO
uv run python src/vae_bernoulli.py evaluate --prior mog --num-components 10 --model vae_mog.pt

# Sample
uv run python src/vae_bernoulli.py sample --prior mog --num-components 10 --model vae_mog.pt --samples samples_mog.png

# Plot prior vs aggregate posterior
uv run python src/vae_bernoulli.py plot_prior_posterior --prior mog --num-components 10 --model vae_mog.pt --prior-posterior-plot prior_posterior_mog.png
```

### 3) VAE with Flow-based prior

```bash
# Train
uv run python src/vae_bernoulli.py train --prior flow --flow-layers 8 --flow-hidden 64 --epochs 10 --model vae_flow.pt

# Evaluate test ELBO
uv run python src/vae_bernoulli.py evaluate --prior flow --flow-layers 8 --flow-hidden 64 --model vae_flow.pt

# Sample
uv run python src/vae_bernoulli.py sample --prior flow --flow-layers 8 --flow-hidden 64 --model vae_flow.pt --samples samples_flow.png

# Plot prior vs aggregate posterior
uv run python src/vae_bernoulli.py plot_prior_posterior --prior flow --flow-layers 8 --flow-hidden 64 --model vae_flow.pt --prior-posterior-plot prior_posterior_flow.png
```

**Note:** The `--prior`, `--num-components`, `--flow-layers`, and `--flow-hidden` flags must match between train/evaluate/sample so the model architecture is reconstructed correctly.

## Part B: Sampling quality (continuous MNIST)

### 4) DDPM using U-Net

```bash
# Train
uv run python src/ddpm.py train --data mnist --batch-size 64 --epochs 10 --model ddpm_mnist.pt

# Sample
uv run python src/ddpm.py sample --data mnist --model ddpm_mnist.pt --samples ddpm_samples.png
```

### 5) Latent DDPM (DDPM in latent space of a beta-VAE)

Three-step workflow: train the beta-VAE, then train the DDPM on encoded latents, then sample.

```bash
# Step 1: Train beta-VAE (Gaussian decoder, continuous MNIST)
uv run python src/latent_ddpm.py train_vae --beta 1e-6 --vae-epochs 10 --vae-model latent_vae_b1e-6.pt

# Step 2: Train DDPM on latent space
uv run python src/latent_ddpm.py train_ddpm --beta 1e-6 --ddpm-epochs 50 --vae-model latent_vae_b1e-6.pt --ddpm-model latent_ddpm_b1e-6.pt

# Step 3: Sample
uv run python src/latent_ddpm.py sample --beta 1e-6 --vae-model latent_vae_b1e-6.pt --ddpm-model latent_ddpm_b1e-6.pt --samples latent_ddpm_samples_b1e-6.png
```

**Note:** The `--beta` flag must match across all three steps. To sweep over beta values for FID comparison:

```bash
for beta in 1e-6 1e-3 1e-1 1.0; do
  uv run python src/latent_ddpm.py train_vae  --beta $beta --vae-model latent_vae_b${beta}.pt
  uv run python src/latent_ddpm.py train_ddpm --beta $beta --vae-model latent_vae_b${beta}.pt --ddpm-model latent_ddpm_b${beta}.pt
  uv run python src/latent_ddpm.py sample     --beta $beta --vae-model latent_vae_b${beta}.pt --ddpm-model latent_ddpm_b${beta}.pt --samples latent_ddpm_samples_b${beta}.png
done
```

## Report

Source: `report/template.tex` (bibliography: `report/template.bib`)

Build from the `report/` directory:

```bash
cd report && latexmk -pdf template.tex
```
