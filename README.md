# Generative AI : Generative variational Recurrent Neural Networks 

> GRU-based variational recurrent models trained to generate synthetic state trajectories from small datasets, with a split-latent variant for handling noisy and noiseless conditions simultaneously.

---

## What this is

Standard VAEs treat each trajectory as a flat vector, ignoring the sequential structure of the data. These two scripts use **Variational Recurrent Neural Networks (VRNNs)** instead, encoding trajectories through a GRU before projecting to the latent space. This lets the model capture temporal dependencies across the 1001-timestep sequences.

Two models are implemented:

| Script | Model | Problem |
|---|---|---|
| `VRNN.py` | Standard VRNN | Threat field evolution data with measurement noise|
| `Split_VRNN.py` | Split-VRNN | Threat field evolution data with measurement noise + Estimated Threat field evolution data |

---

## Background

Each trajectory is a time series of shape `[1001 timesteps, 100 state features]` from an evolving threat field. The VRNN encodes the full sequence into a latent vector using a GRU, then decodes it back to the original sequence length. The Split-VRNN extends this by separating the latent space into two parts: z1 captures condition-specific variation (noisy(with measurement noise) vs. noiseless/estimated) and z2 captures shared dynamics across both.

- **VRNN training set**: 10 noisy trajectories (`set09`)
- **Split-VRNN training set**: 25 noisy (`set05`) + 25 noiseless (`realcase_set05`)

Both models generate 1000 new trajectories after training.

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
---

## Hyperparameters

| Parameter | VRNN | Split-VRNN |
|---|---|---|
| `input_dim` | 10 | 10 |
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


---

