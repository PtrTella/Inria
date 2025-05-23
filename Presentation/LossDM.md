---
marp: true
math: mathjax
---
# Deriving the Objective Function from  

**Deep Unsupervised Learning using Nonequilibrium Thermodynamics** and **Denoising Diffusion Probabilistic Models**

This derivation aims to express the log-likelihood of a generative model via nonequilibrium thermodynamic principles and estimate it using importance sampling over stochastic trajectories.

---

## **Objective: Log-Likelihood of the Data**

Given a data distribution $q(x_0)$ (typically an empirical distribution from the dataset), the goal is to maximize the expected log-likelihood under a generative model $p_\theta(x_0)$:

$$
\mathcal{L_\theta} = \mathbb{E}_{q(x_0)} \left[ \log p_\theta(x_0) \right]
$$

---

## **Intractability of the Marginal Likelihood**

The model defines a joint distribution over a trajectory $x_{0:T} = (x_0, x_1, \dots, x_T)$. The marginal likelihood of $x_0$ is:

$$\
p_\theta(x_0) = \int p_\theta(x_{0:T}) \, dx_{1:T}
$$

where the joint is decomposed as a **reverse process**:

$$
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)
$$

Here:
- $p(x_T)$ is a simple prior (e.g., standard Gaussian),
- $p_\theta(x_{t-1} \mid x_t)$ is the learned generative model (parameterized by a neural network).

---

## **Importance Sampling via a Forward Process**

The marginal likelihood $p_\theta(x_0)$ is intractable due to the high-dimensional integral over all possible trajectories $x_{1:T}$. To estimate it, we apply **importance sampling**, a technique that lets us rewrite an expectation over a difficult distribution $p$ using samples from a simpler, tractable distribution $q$.

We define a tractable **forward (proposal) process** $q(x_{1:T} \mid x_0)$, and use it to rewrite:

$$
p_\theta(x_0) = \int q(x_{1:T} \mid x_0) \cdot \frac{p_\theta(x_{0:T}) }{q(x_{1:T} \mid x_0)} \, dx_{1:T}
$$

Taking the expectation over $q$:

$$
p_\theta(x_0) 
= \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \frac{p_\theta(x_{0:T}) }{q(x_{1:T} \mid x_0)} \right] 
= \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \frac{p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)}{\prod_{t=1}^T q(x_t | x_{t-1})} \right] 
$$

---

### Intuition

- $p(x_T) \prod p_\theta(x_{t-1} \mid x_t)$ is the **target distribution**, representing the reverse generative process from noise to data.
- $q(x_{1:T} \mid x_0)$ is the **proposal distribution**, a forward process (e.g. a noising or diffusion process) that we can sample from easily.

Importance sampling allows us to evaluate expectations or integrals under $p$ using samples from $q$, correcting for the mismatch via the ratio $\frac{p}{q}$. This is especially useful when sampling directly from $p$ is hard or intractable, as is the case here.

---

## **Lower Bound via Jensen’s Inequality**

Now plug this into the expected log-likelihood, importance sampling on $p_\theta(x_0)$:

$$
\mathcal{L_\theta} = \mathbb{E}_{q(x_0)} \left[ \log p_\theta(x_0) \right]
= \mathbb{E}_{q(x_0)} \left[ \log \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \frac{p_\theta(x_{0:T}) }{q(x_{1:T} \mid x_0)} \right] \right]
$$

Apply **Jensen's inequality**

$$\mathcal{L_\theta} \geq \mathcal{K} =  \mathbb{E}_{q(x_0)} \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \log \left( \frac{p_\theta(x_{0:T}) }{q(x_{1:T} \mid x_0)} \right) \right]
$$

---

## **Defining the ELBO (Evidence Lower Bound)**

We define the **joint forward distribution** : $q(x_{0:T}) = q(x_0) q(x_{1:T} \mid x_0)$

Thus, the lower bound becomes:

$$
\mathcal{L_\theta} \geq \mathcal{K} = \mathbb{E}_{q(x_{0:T})} \left[ \log \left( \frac{p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)}{q(x_{1:T} \mid x_0)} \right) \right]
$$

Expanding the expectation:
$$
\mathcal{L_\theta} \geq \mathcal{K} = \int q(x_{0:T}) \log \left( \frac{p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)}{q(x_{1:T} \mid x_0)} \right) dx_{0:T}
$$

---

![image](./images/Derivation.png)

---

# 💠 KL Divergence formula
We obtain the following expression based on [Ho et al. 2020](./ElboToKLDinHO.html) derivation, but similarly the original computation from [Sohl-Dickstein et al. 2015](./ElboToKLD.html) leads to the similar results.

$$
\mathcal{L}_{\text{vlb}} = \mathbb{E}_q \left[
\underbrace{D_{\text{KL}}(q(x_T|x_0) \| p(x_T))}_{L_T}
+ \sum_{t=2}^T \underbrace{D_{\text{KL}}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}}
- \underbrace{\log p_\theta(x_0|x_1)}_{L_0}
\right]
$$

---

## First Term - Prior KL
$$L_T = D_{\text{KL}}(q(x_T|x_0) \| p(x_T))$$

Quantify how much the final noise $x_T$ (resulting from the forward process) resembles the standard Gaussian noise distribution  $p(x_T)$.

In practice, since the **forward process is very long** (small noise additions $\beta_t$), at the end, $q(x_T|x_0)$ becomes **almost identical** to $p(x_T)$. This KL divergence becomes **very small** and can be **ignored** (or computed once and treated as constant).

---

## Third Term - Reconstruction term
$$L_0 = -\log p_\theta(x_0 | x_1)$$

At the final step, we need to model the probability of the original discrete data $x_0$ (like pixel values 0–255) starting from a slightly noised version $x_1$. Since the model works in a continuous space $[-1, 1]$, we define the last reverse step $p_\theta(x_0|x_1)$ by discretizing a simple Gaussian distribution:

$$
p_\theta(x_0 | x_1) = \prod_{i=1}^D \int_{\delta^-(x_0^i)}^{\delta^+(x_0^i)} \mathcal{N}(x; \mu^i_\theta(x_1, 1), \sigma^2_1) \, dx
$$
---

- $x_0 \in \{0, 1, \ldots, 255\}^D$ is the quantized data (e.g., RGB pixel values),
- $\mu_\theta^i(x_1, 1)$ is the predicted mean for pixel $i$,
- $\sigma_1^2$ is the fixed variance used by the model at the first iteration of the reverse process (typically $\beta_1$ or $\tilde{\beta}_1$).


> ⚠️ The reconstruction term $L_0$ is not used in the standard training loss of Ho et al. (2020). It is replaced by a simpler loss, $L_{\text{simple}}$, which trains the model to predict the noise added to the data. This approach has shown better performance in terms of the visual quality of the generated samples.

---

## Second Term — Transition KLs

We consider the term:

$$
\sum_{t=2}^T D_{\text{KL}}\left(q(x_{t-1} \mid x_t, x_0) \, \| \, p_\theta(x_{t-1} \mid x_t)\right)
$$

This measures how well the learned reverse process $p_\theta(x_{t-1} \mid x_t)$ approximates the **true posterior** distribution over the intermediate state $x_{t-1}$, given $x_t$ and the original data $x_0$. This posterior is analytically tractable due to the Gaussianity of the forward process.

---

## **Computing the True Posterior $q(x_{t-1} \mid x_t, x_0)$**

Given that all transitions in the forward process are Gaussian, and that $x_t$ is a linear transformation of $x_0$ plus Gaussian noise, the posterior is also Gaussian:

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\left(\tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I\right)
$$

Where:

* The **posterior mean** is:

$$
\tilde{\mu}_t(x_t, x_0) = 
\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 
+ \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t
$$

* The **posterior variance** is:

$$
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$

---

## **KL Between Gaussians**

The KL divergence between the posterior and the model prediction — both Gaussians with diagonal covariance — is:

$$
D_{\text{KL}}(q \, \| \, p) = \frac{1}{2}\left[
\log \frac{\sigma_\theta^2}{\tilde{\beta}_t} +
\frac{\tilde{\beta}_t}{\sigma_\theta^2} +
\frac{\left\| \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \right\|^2}{\sigma_\theta^2}
- 1
\right]
$$

Where:

* $\tilde{\mu}_t(x_t, x_0)$ is the posterior mean,
* $\mu_\theta(x_t, t)$ is the model’s predicted mean,
* $\sigma_\theta^2$ is the variance of the reverse model, which is either fixed or learned.

---

## **Simplification: Fixing the Variance**

Following *Ho et al.*'s approach, if we **fix the model variance** to match the posterior, $\sigma_\theta^2 = \tilde{\beta}_t$, then:

* The log-ratio and variance terms cancel or simplify,
* The KL becomes proportional to the squared error between predicted and true means:

$$
D_{\text{KL}}(q \, \| \, p) = \frac{1}{2 \tilde{\beta}_t}
\left\| \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \right\|^2
$$

---

## 💠 Mean Prediction Loss

This suggests the following training objective:

$$
\mathcal{L}_{\text{mean}} = \mathbb{E}_q \left[
\left\| \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \right\|^2
\right]
$$

However, *Ho et al.* propose a further simplification — rather than predicting $\mu_\theta$, they predict the **original noise** $\epsilon$ used to generate $x_t$.

---

### 🔁 Rewriting the Forward Process

From the forward process:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Solving for $x_0$:

$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon \right)
$$

Plugging this into $\tilde{\mu}_t(x_t, x_0)$, both the posterior and model means become linear combinations of $x_t$ and $\epsilon$ or $\epsilon_\theta$.

---

### ⚡ Subtracting Posterior and Model Means

If we write both means:

$$
\tilde{\mu}_t(x_t, \epsilon) = A_t x_t + B_t \epsilon \\
\mu_\theta(x_t, t) = A_t x_t + B_t \epsilon_\theta(x_t, t)
$$

Then:

$$
\tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) = B_t (\epsilon - \epsilon_\theta(x_t, t))
$$

So minimizing the KL is equivalent to minimizing the error in noise prediction.

---

## 💠 Final Training Objective

This yields the simplified loss:

$$
\mathcal{L}_{\text{simple}} =
\mathbb{E}_{x_0, t, \epsilon} \left[
\left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2
\right]
$$

Where:

* $x_0 \sim \text{data}$,
* $t \sim \text{Uniform}(1, T)$,
* $\epsilon \sim \mathcal{N}(0, I)$,
* $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$

---

### ✅ Interpretation

The model is trained to predict the exact noise $\epsilon$ used in the forward process — this is a **simple, efficient, and empirically effective** surrogate for minimizing the full variational bound.

It also leads to **clean analytical properties**, enabling high-quality sampling with relatively few steps.

---

Let me know if you'd like this cleaned into a full PDF section or adapted to a presentation!


## References

- Ho, J., Jain, A., & Abbeel, P. (2020). **Denoising Diffusion Probabilistic Models**. *Advances in Neural Information Processing Systems (NeurIPS)*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). **Deep Unsupervised Learning using Nonequilibrium Thermodynamics**. *International Conference on Machine Learning (ICML)*. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585)

- Nichol, A. Q., & Dhariwal, P. (2021). **Improved Denoising Diffusion Probabilistic Models**. *International Conference on Machine Learning (ICML)*. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)