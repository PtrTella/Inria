---
marp: true
math: mathjax
---

# Forward and Reverse Processes in Diffusion Models

Diffusion models, as introduced by Sohl-Dickstein et al. (2015) and further developed by Ho et al. (2020), Nichol and Dhariwal (2021), and others, rely on two key processes: **Forward Process (Diffusion Process)** and **Reverse Process (Generative Process)**. These processes form the core of how these models operate, facilitating both the gradual destruction of data (forward) and its reconstruction (reverse).

---

## Forward Process (Diffusion Process)

The forward process is a **Markovian noising process** that incrementally adds noise to a data sample $x_0$ over a series of $T$ timesteps, transforming it into a purely noisy distribution $x_T$. The intuition behind this process is to gradually corrupt the data distribution until it approximates a simple distribution, typically an **isotropic Gaussian**.

---

### Mathematical Definition:

Given an initial data sample $x_0 \sim q(x_0)$, the forward process defines a sequence of latent variables $x_1, x_2, \dots, x_T$ as follows:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I)
$$

where:

* $\beta_t \in (0, 1)$ is the noise variance schedule for each timestep $t$.
* The mean term $\sqrt{1 - \beta_t}x_{t-1}$ scales down the previous state, while $\beta_t I$ adds Gaussian noise.

---

### Marginalization:

Importantly, the forward process allows direct sampling of $x_t$ at any timestep $t$ conditioned on $x_0$:

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha_t}}x_0, (1 - \bar{\alpha_t})I)
$$

where:

* $\alpha_t = 1 - \beta_t$
* $\bar{\alpha_t} = \prod_{s=1}^t \alpha_s$

This expression indicates that $x_t$ is a Gaussian perturbation of $x_0$, controlled by the variance schedule.

---

## Reverse Process (Generative Process)

The reverse process aims to **reconstruct $x_0$ from $x_T$** by iteratively denoising the sample. Conceptually, this is a learned approximation of the reverse diffusion:

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

where $\mu_\theta$ and $\Sigma_\theta$ are neural networks that predict the mean and variance of the denoised state, respectively.

---

### Mean Estimation:

Ho et al. (2020) proposed parameterizing the mean as:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha_t}}} \epsilon_\theta(x_t, t) \right)
$$

where $\epsilon_\theta(x_t, t)$ is a network trained to predict the noise component of $x_t$.

### Variance Estimation:

The variance $\Sigma_\theta(x_t, t)$ can be:

* Fixed to a known schedule (Ho et al., 2020), or
* Learned as a separate neural network head (Nichol and Dhariwal, 2021).

---

## Summary:

* The **forward process** destroys information gradually, turning structured data into noise.
* The **reverse process** learns to reverse this destruction, reconstructing the original data sample through a series of denoising steps.

In essence, the forward and reverse processes are two sides of the same coin: one destroys structure, and the other reconstructs it through learned denoising transitions.

Diffusion models have demonstrated state-of-the-art results in image synthesis (Dhariwal and Nichol, 2021), outperforming GANs in several quality metrics. Their well-defined forward and reverse processes allow for high-quality sample generation, as well as smoother convergence during training.
