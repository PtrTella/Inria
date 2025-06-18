---
marp: true
math: mathjax
---

# Improvements in DDPMs
This document explores the key advances proposed in *Improved Denoising Diffusion Probabilistic Models* (Nichol & Dhariwal, 2021).

---

## Learning Reverse Variance $\Sigma_\theta$

Instead of fixing variance, the model **learns a log-interpolated variance** between two bounds:

$$
\Sigma_\theta(x_t, t) = \exp \left( v \log \beta_t + (1 - v) \log \tilde{\beta}_t \right)
$$

- $v \in [0, 1]$ is predicted by the network.
- This keeps $\Sigma_\theta$ in a numerically stable range.

> In practise they do not apply any constraints on $v$, allowing the model to predict it freely, but they didn't observe network doing it, so the bounds are expressive enough.

[Variance IMG](./images/Variance.png)

---

## Optimizing the Log-Likelihood

Traditional DDPMs optimize a simplified loss. Here, the authors propose a **hybrid objective**:

$$
\mathcal{L}_{\text{hybrid}} = \mathcal{L}_{\text{simple}} + \lambda \mathcal{L}_{\text{vlb}}
$$

- $\mathcal{L}_{\text{simple}}$: weighted MSE loss on predicted noise $\epsilon_\theta$.
- $\mathcal{L}_{\text{vlb}}$: variational lower bound (exact KL terms).
- $\lambda = 0.001$: balances sample quality and likelihood.

> This hybrid form stabilizes training and allows learning variance $\Sigma_\theta(x_t, t)$ without deteriorating sample quality.

---

## Cosine Noise Schedule

Linear $\beta_t$ schedules over-diffuse early. Instead, define $\bar{\alpha}_t$ via a cosine-based function to have a linear drop-off in the middle of the process (and a slower near to extremes):

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)
$$

where $\beta_t = 1 - \bar{\alpha}_t / \bar{\alpha}_{t-1}$. The offset $s$ has been defined to avoid too small $\beta_t$ values near $t = 0$ since they found out that it makes hard for the model to estimate $\epsilon$ in the first steps.

[Cosine Schedule IMG](./images/CosScheduler.png)

---

## Reducing Gradient Noise via Importance Sampling

The variance of $\mathcal{L}_{\text{vlb}}$ is large due to uneven contribution across $t$. So, the authors apply **importance sampling**:

$$
\mathcal{L}_{\text{vlb}} = \mathbb{E}_{t \sim p_t} \left[ \frac{L_t}{p_t} \right], \quad p_t \propto \sqrt{\mathbb{E}[L_t^2]}
$$

This allows the model to focus on more informative timesteps. In the paper the presample each $t$ 10 times uniformly, then sample $t$ from the resulting distribution.

[Importance Sampling IMG](./images/ImportanceSampling.png)

---

## Fast Sampling via Fewer Steps

After training with $T = 4000$, samples can be generated with $K \ll T$ steps:

1. Choose $K$ timesteps $\{t_1, ..., t_K\}$ by **uniform striding**.
2. Use $\bar{\alpha}_{t_i}$ to rederive $β_{t_i}$ and $\tilde{β}_{t_i}$.
3. Predict $\mu_\theta$ and $\Sigma_\theta$ at these reduced steps.

> With 50–100 steps, models maintain near-optimal FID.

[Fast Sampling IMG](./images/FastSampling.png)

---

## References

- Nichol, A., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
