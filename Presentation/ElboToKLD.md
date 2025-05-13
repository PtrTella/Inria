---
marp: true
math: mathjax
--- 

## **Log Likelihood Lower Bound (ELBO)**

The starting point is the Evidence Lower Bound (ELBO), expressed as follows:

$$
\mathcal{L} \geq 
K = \int q(x_{0:T}) \log \left[ \frac{p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)}{q(x_{1:T} \mid x_0)} \right] dx_{0:T}
$$

This is the **Evidence Lower Bound (ELBO)**. It represents a lower bound on the log-likelihood of the model. Let's break down each term:

* $q(x_{0:T})$ is the forward process (diffusion process).
* $p(x_T)$ is the prior distribution on the noised state $x_T$.
* $p_\theta(x_{t-1} | x_t)$ is the reverse process (denoising process).
* $q(x_{1:T} | x_0)$ is the forward diffusion trajectory conditioned on $x_0$.

---

## 🔍 **Understanding Entropy in the Diffusion Process**

Entropy, in this context, measures the **uncertainty** or **disorder** of the distribution $p(x_T)$. Formally, the entropy of a distribution $p(x)$ is defined as:

$$
H(p) = -\int p(x) \log p(x) \, dx
$$

In our derivation, the entropy of $p(x_T)$ represents the **maximum noise level** reached during the forward diffusion process. At this stage $x_T$, all information is fully diffused and resembles a Gaussian distribution with maximum randomness.

---

## **Splitting the Log-Likelihood Term**

The next step is a crucial manipulation: **we separate the term involving $p(x_T)$**. This is possible because the integral over time $T$ can be isolated, rewriting the expression as:

$$
K = \int q(x_{0:T}) \sum_{t=1}^T \log \left[\frac{p_\theta(x_{t-1} | x_t)}{q(x_t | x_{t-1})}\right] dx_{0:T} + \int q(x_T) \log p(x_T) \, dx_T
$$

This is where **entropy** is introduced, and represents the **uncertainty** of the final diffused state in the forward process.

$$
\int q(x_T) \log p(x_T) \, dx_T = -H(p(x_T))
$$

---


## **Removing Edge Effects**

In the next step, to **remove edge effects** at time $t = 0$, the following assumption is made:

$$
p_\theta(x_0 | x_1) = q(x_1 | x_0) \frac{\pi(x_0)}{\pi(x_1)}
$$

where $\pi(x_0)$ and $\pi(x_1)$ are reference distributions in the forward process. This assumption allows the cancellation of the initial term, so the summation in the integral starts from $t = 2$:

$$
K = \sum_{t=2}^T \int dx_{0:T} \, q(x_{0:T}) \log \left[\frac{p_\theta(x_{t-1} | x_t)}{q(x_t | x_{t-1})}\right] - H(p(x_T))
$$

---

## **Rewriting in Terms of Posterior Distributions**

To further simplify, we observe that the **forward diffusion process** is a Markov chain. This allows us to express it in terms of **posterior distributions**:

$$
q(x_t | x_{t-1}) = \int q(x_t | x_{t-1}, x_0) q(x_0) \, dx_0
$$

The posterior conditioned on the starting state $x_0$ can be rewritten as:

$$
q(x_t | x_{t-1}) = q(x_t | x_0) \cdot \frac{q(x_{t-1} | x_t, x_0)}{q(x_{t-1} | x_0)}
$$

This decomposition allows us to inject **Bayes' Rule** to the logarithm term.

---

## 🔍 **Applying Bayes' Rule**

We apply Bayes' Rule:

$$
q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}) q(x_{t-1} | x_0)}{q(x_t | x_0)}
$$

Substituting this back into the expression for $K$:

$$
K = \sum_{t=2}^T \int q(x_{0:T}) \log \left[ \frac{p_\theta(x_{t-1} | x_t)}{q(x_{t-1} | x_t, x_0)} \cdot \frac{q(x_{t-1} | x_0)}{q(x_t | x_0)} \right] dx_{0:T} - H(p(x_T))
$$

Now, the expression inside the logarithm is split into:

1. A **likelihood ratio**: $\frac{p_\theta(x_{t-1} | x_t)}{q(x_{t-1} | x_t, x_0)}$
2. A **prior ratio**: $\frac{q(x_{t-1} | x_0)}{q(x_t | x_0)}$

---

## 🔍 **Introducing Conditional Entropies**

The paper then recognizes that several terms can be expressed as **conditional entropies**. Let's rewrite it explicitly:

$$
K = \sum_{t=2}^T \int q(x_{0:T}) \log \left[ \frac{p_\theta(x_{t-1} | x_t)}{q(x_{t-1} | x_t, x_0)} \right] dx_{0:T} + \sum_{t=2}^T \left[ H_q(X_t | X_0) - H_q(X_{t-1} | X_0) \right] - H(p(x_T))
$$

Here:

* $H_q(X_t | X_0)$ represents the conditional entropy at time $t$ given the initial state.
* The differences between these entropies reflect how much uncertainty is introduced as the diffusion progresses.

---

## 🔍 **Final Rewriting in Terms of KL Divergences**

Finally, the expression can be transformed neatly into **KL Divergences**. Recall that the KL Divergence measures the difference between the true forward diffusion and the reverse generative process:

$$
K = - \sum_{t=2}^T \int q(x_0, x_t) D_{KL}\left(q(x_{t-1} | x_t, x_0) \, \Big|\Big| \, p_\theta(x_{t-1} | x_t)\right) dx_0 \, dx_t \, + H_q(X_T | X_0) - H_q(X_1 | X_0) - H(p(x_T))
$$

---

## 🔍 **Summary of the Terms**

1. The main term is a **sum of KL divergences** for every step $t$ from 2 to $T$.
2. The second and third terms are **conditional entropies** that represent the change in uncertainty.
3. The final term is the **entropy of the final distribution** $p(x_T)$, representing the maximum noise.

---

### 🚀 **Why is this important?**

The purpose of this manipulation is to **decompose the total uncertainty** into:

* A sum of mismatches (KL Divergences) between the forward process and the reverse generative process.
* Entropy terms that represent the randomness introduced during the forward diffusion.

---

Would you like me to move forward and explain the **role of these entropies in simplifying the final objective**? Or is there any step you want me to go deeper into?


### 🚀 **Why did they do this?**

The main idea is to decompose the entire process into:

* How well the reverse process matches the true forward diffusion (KL Divergence terms).
* The uncertainty introduced at each step (entropy terms).

---

Would you like me to explain now **why the entropy terms simplify** and what role they play in the final loss? This is a crucial step to understand why the KL Divergence is the final expression.


## 🔍 **5. Rewriting in terms of KL Divergences**

Finally, the expression is rewritten in terms of **KL Divergences**. Recall that the KL Divergence between two distributions $q$ and $p$ is defined as:

$$
D_{KL}(q || p) = \int q(x) \log \frac{q(x)}{p(x)} \, dx
$$

The final expression becomes:

$$
K = -\sum_{t=2}^T \int dx_0 \, dx_t \, q(x_0, x_t) D_{KL}\left(q(x_{t-1} | x_t, x_0) || p_\theta(x_{t-1} | x_t)\right) + H(q(X_T | X_0)) - H(q(X_1 | X_0)) - H(p(X_T))
$$

---

### 🚀 **Why does entropy simplify things?**

When we consider the entropy of the forward process, we are essentially saying: *"I already know how much noise I have introduced into the system"*. This value is a constant that can be calculated, which simplifies the expressions because we do not need to track every intermediate step of diffusion.

---

Would you like me to continue with the final part where they change the integrals into KL divergences using Bayes' rule and complete the full derivation? This is the part where the formula transforms into the familiar KL divergence sum form.
