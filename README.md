# Generative AI : Variational Recurrent Neural Networks (VRNN & Split-VRNN)

> GRU-based variational recurrent models trained to generate synthetic state trajectories from small datasets, with a split-latent variant for handling noisy and noiseless conditions simultaneously.

---

## What this is

Standard VAEs treat each trajectory as a flat vector, ignoring the sequential structure of the data. These two scripts use **Variational Recurrent Neural Networks (VRNNs)** instead, encoding trajectories through a GRU before projecting to the latent space. This lets the model capture temporal dependencies across the 1001-timestep sequences.

Two models are implemented:

| Script | Model | Problem |
|---|---|---|
| `VRNN.py` | Standard VRNN | LTI system (noisy trajectories only) |
| `Split_VRNN.py` | Split-VRNN | LTI system (noisy + noiseless, disentangled latent) |

---

## Background

Each trajectory is a time series of shape `[1001 timesteps, 100 state features]` from an LTI dynamical system. The VRNN encodes the full sequence into a latent vector using a GRU, then decodes it back to the original sequence length. The Split-VRNN extends this by separating the latent space into two parts: z1 captures condition-specific variation (noisy vs. noiseless) and z2 captures shared dynamics across both.

- **VRNN training set**: 10 noisy trajectories (`set09`)
- **Split-VRNN training set**: 25 noisy (`set05`) + 25 noiseless (`realcase_set05`)

Both models generate 1000 new trajectories after training, saved as individual CSV files.

---

## Model Architectures

### VRNN (`VRNN.py`)

A single-domain variational recurrent model. The encoder GRU processes the full input sequence and its final hidden state is projected to `[mu, logvar]`. The decoder GRU takes the sampled latent vector, repeated across all timesteps, and reconstructs the sequence.

```
Input [batch, 1001, 100]
    |
Encoder GRU -> final hidden state h
    |
LayerNorm -> ReLU -> FC
    |
[mu (16-dim), logvar (16-dim)]
    |
Reparameterize: z = mu + eps * sigma,  eps ~ N(0, I)
    |
z repeated across 1001 timesteps -> Decoder GRU
    |
LayerNorm -> ReLU -> FC
    |
Output [batch, 1001, 100]
```

**Loss:**
```
L = MSE(x_hat, x)  +  KL[ q(z|x) || N(0, I) ]
```

---

### Split-VRNN (`Split_VRNN.py`)

Same GRU-based encoder-decoder structure, but the encoder projects to two separate latent heads: z1 (condition-specific) and z2 (shared). Both are concatenated before being passed to the decoder. A regularization penalty on z1 for noisy samples pushes condition-specific variation into z1 and shared dynamics into z2.

```
Input [batch, 1001, 10]  +  label (0 = noisy, 1 = noiseless)
    |
Encoder GRU -> final hidden state h
    |
LayerNorm -> ReLU -> FC
    |
[z1_mu, z1_logvar (16-dim)]   [z2_mu, z2_logvar (16-dim)]
    |                                  |
Reparameterize z1              Reparameterize z2
    |                                  |
         cat([z1, z2]) repeated x 1001 timesteps
                    |
             Decoder GRU
                    |
             LayerNorm -> FC
                    |
         Output [batch, 1001, 10]
```

**Loss:**
```
L = MSE(x_hat, x)
  + KL[q(z1) || N(0, I)]
  + KL[q(z2) || N(0, I)]
  + lambda_reg * ||z1||^2   (applied only to noisy samples)
```

`lambda_reg = 1.0`. Note the regularization direction is inverted relative to the Split-VAE: here the penalty targets noisy samples (label = 0) rather than noiseless ones, pushing noise-related variation into z1.

---

## Hyperparameters

| Parameter | VRNN | Split-VRNN |
|---|---|---|
| `input_dim` | 100 | 10 |
| `hidden_dim` | 40 | 40 |
| `latent_dim` / `z1_dim + z2_dim` | 16 | 16 + 16 |
| `seq_len` | 1001 | 1001 |
| `epochs` | 2000 | 1000 |
| `batch_size` | 32 | 32 |
| `learning_rate` | 1e-3 | 1e-3 |
| `lambda_reg` | n/a | 1.0 |
| Training samples | 10 noisy | 25 noisy + 25 noiseless |
| Generated samples | 1000 | 1000 |

The VRNN uses 100-feature trajectories; the Split-VRNN uses 10-feature trajectories. Both share the same GRU hidden size and latent dimensionality.

---

## Installation

```bash
git clone https://github.com/NachiketBa/Generative-AI-for-RL.git
cd "Generative-AI-for-RL"

pip install torch pandas numpy
```

Tested on Python 3.9+ and PyTorch 2.0+. Both scripts detect CUDA automatically and fall back to CPU if no GPU is found.

---

## Data Format

Each dataset is a folder of CSV files, one file per trajectory.

```
case01/set09/
    traj_0001.csv      # shape on disk: [100, 1001] -> transposed to [1001, 100]
    traj_0002.csv
    ...

case01/set05/
    traj_0001.csv      # shape on disk: [10, 1001] -> transposed to [1001, 10]
    ...

realcase_set05/
    realtraj_0001.csv
    ...
```

Each CSV stores a trajectory as a matrix with features along rows and timesteps along columns. Both scripts transpose on load so the final tensor shape is `[num_samples, 1001, num_features]`.

> **Update all hardcoded folder paths** near the top of each script before running.

---

## Running the scripts

### VRNN

```bash
python VRNN.py
```

Loads the first 10 trajectories from `set09`, trains for 2000 epochs, then generates 1000 samples. Each sample is saved as a separate CSV file (shape: `[100, 1001]`, transposed back on save) in:

```
generated_100_vrnn_states_10_new/
    generated_sample_0000.csv
    ...
    generated_sample_0999.csv
```

### Split-VRNN

```bash
python Split_VRNN.py
```

Loads the first 25 noisy trajectories from `set05` and the first 25 noiseless trajectories from `realcase_set05`, trains for 1000 epochs, then generates 1000 samples from random z1 and z2 draws. Each sample is saved as a separate CSV file (shape: `[10, 1001]`, transposed back on save) in:

```
generated_10_splitvrnn_states_25_1/
    generated_sample_0000.csv
    ...
    generated_sample_0999.csv
```

---

## Console output

**VRNN** prints one line per epoch:
```
Epoch [1/2000], Loss: 3241.8823
Epoch [2/2000], Loss: 2987.4410
```

**Split-VRNN** prints one line per epoch:
```
Epoch 1: Loss = 4182.3301
Epoch 2: Loss = 3894.7712
```

Both print a confirmation line when generation is complete:
```
Generated 1000 samples saved as CSV files in <save_dir>.
```

---

## Design notes

**GRU over flat input.** The VAE scripts in this repo flatten each trajectory to a single vector. The VRNN models feed the sequence timestep-by-timestep into a GRU, so the encoder sees the full temporal structure before projecting to the latent space. This matters for LTI trajectories where the dynamics at one timestep depend on the previous one.

**LayerNorm on GRU output.** Both models apply LayerNorm to the GRU's final hidden state before the linear projection. With only 10 or 25 training trajectories, batch statistics are too noisy to normalize reliably, so LayerNorm normalizes per sample instead.

**Latent vector repeated across timesteps.** The decoder takes a single latent vector z and repeats it across all 1001 timesteps before feeding it into the decoder GRU. This forces the latent space to encode global trajectory shape rather than per-timestep variation, keeping the generation process simple — sample one z, get one full trajectory.

**Regularization targets noisy samples in Split-VRNN.** In the Split-VAE scripts elsewhere in this repo, the penalty targets noiseless samples to push z1 toward zero for clean data. Here the logic is inverted: the penalty targets noisy samples (label = 0), pushing noise-related variation into z1 and leaving z2 to capture the shared clean dynamics.

**Very small training sets.** The VRNN trains on 10 trajectories and the Split-VRNN on 25 per domain. Both models are intentionally lightweight (hidden_dim = 40, latent_dim = 16) to avoid overfitting at this scale.

---

## File structure

```
Generative-AI-for-RL/
    Mars Lander Problem/
        S_VAE_mars_lander.py
        MI_VAE_mars_lander.py
        README.md
    LTI Problem/
        S_VAE_LTI.py
        Split_VAE_LTI.py
        VRNN.py
        Split_VRNN.py
    Minthreat Problem/
        S_VAE_minthreat.py
        Split_VAE_minthreat.py
    Zermelo Navigation Problem/
        S_VAE_zermelo.py
        Z_VAE_zermelo.py
        SGAN_zermelo.py
        ZGAN_ham.py
        ZGAN_ham_head.py
    README.md
```

---

## Citation

```bibtex
@misc{nachiket2025genairl,
  author       = {Nachiket Ba},
  title        = {Generative AI for Reinforcement Learning},
  year         = {2025},
  howpublished = {\url{https://github.com/NachiketBa/Generative-AI-for-RL}},
}
```

---

## License

MIT License.
