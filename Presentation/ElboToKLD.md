---
marp: true
math: mathjax
--- 

## **Log Likelihood Lower Bound (ELBO)**

The starting point is the Evidence Lower Bound (ELBO), expressed as follows:

$$
\mathcal{K} 
= \int q(x_{0:T}) \log \left( \frac{p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)}{q(x_{1:T} \mid x_0)} \right) dx_{0:T}
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
\mathcal{K}
= \int q(x_{0:T}) \sum_{t=1}^T \log \left(\frac{p_\theta(x_{t-1} | x_t)}{q(x_t | x_{t-1})}\right) dx_{0:T} + \int q(x_T) \log p(x_T) \, dx_T
$$

This is where **entropy** is introduced, and represents the **uncertainty** of the final diffused state in the forward process.

$$
\int q(x_T) \log p(x_T) \, dx_T = -H(p(x_T))
$$

---


## **Removing Edge Effects**

To **remove edge effects** at time $t = 0$, Sohl-Dickstein et al. set the final step of the reverse trajectory to be identical to the corresponding forward diffusion step:

$$
p_\theta(x_0 | x_1) = q(x_1 | x_0) \frac{\pi(x_0)}{\pi(x_1)}
$$

> Da controllare perche assunzione e che pi sia uguale senno che senso ha? comunque idea di base e che allo step 1 il rumore aggiunto e trascurabile.

where $\pi(x_0)$ and $\pi(x_1)$ are reference distributions in the forward process. This assumption allows the cancellation of the initial term, so the summation in the integral starts from $t = 2$:

$$
K = \sum_{t=2}^T \int dx_{0:T} \, q(x_{0:T}) \log \left(\frac{p_\theta(x_{t-1} | x_t)}{q(x_t | x_{t-1})}\right) - H(p(x_T))
$$

---

## **Rewriting in Terms of Posterior Distributions**

To further simplify, we observe that the **forward diffusion process** is a Markov chain. This allows us to express it in terms of **posterior distributions**:

$$
q(x_t | x_{t-1}) = \int q(x_t | x_{t-1}, x_0) q(x_0) \, dx_0
$$

The posterior conditioned on the starting state $x_0$ can be rewritten (**Bayes's Rule**):

$$
q(x_t | x_{t-1}) = q(x_t | x_0) \cdot \frac{q(x_{t-1} | x_t, x_0)}{q(x_{t-1} | x_0)} \rightarrow
q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}) q(x_{t-1} | x_0)}{q(x_t | x_0)}
$$

---

Substituting this back into the expression for $K$:

$$
K = \sum_{t=2}^T \int q(x_{0:T}) \log \left( \frac{p_\theta(x_{t-1} | x_t)}{q(x_{t-1} | x_t, x_0)} \cdot \frac{q(x_{t-1} | x_0)}{q(x_t | x_0)} \right) dx_{0:T} - H_p(x_T)
$$

Now, the expression inside the logarithm is split into:

1. A **likelihood ratio**: $\frac{p_\theta(x_{t-1} | x_t)}{q(x_{t-1} | x_t, x_0)}$
2. A **prior ratio**: $\frac{q(x_{t-1} | x_0)}{q(x_t | x_0)}$

---

## **Introducing Conditional Entropies**

The paper then recognizes that several terms can be expressed as **conditional entropies**. Let's rewrite it explicitly:

$$
K = \sum_{t=2}^T \int q(x_{0:T}) \log \left( \frac{p_\theta(x_{t-1} | x_t)}{q(x_{t-1} | x_t, x_0)} \right) dx_{0:T} + \sum_{t=2}^T \left[ H_q(X_t | X_0) - H_q(X_{t-1} | X_0) \right] - H_p(x_T)
$$

Here:

* $H_q(X_t | X_0)$ represents the conditional entropy at time $t$ given the initial state.
* The differences between these entropies reflect how much uncertainty is introduced as the diffusion progresses.

---

## **Rewriting in terms of KL Divergences**

Finally, the expression is rewritten in terms of **KL Divergences**. Recall that the KL Divergence between two distributions $q$ and $p$ is defined as:

$$
D_{KL}(q || p) = \int q(x) \log \frac{q(x)}{p(x)} \, dx
$$

The final expression becomes (minus from reverse log ratio):

$$
K = - \sum_{t=2}^T \int dx_0 \, dx_t \, q(x_0, x_t) D_{KL}\left(q(x_{t-1} | x_t, x_0) || p_\theta(x_{t-1} | x_t)\right) + H_q(X_T | X_0) - H_q(X_1 | X_0) - H_p(X_T)
$$

---

## **Notes**
The **main term** is a **sum of KL divergences** for every step $t$ from 2 to $T$. The second and third terms are **conditional entropies** that represent the change in uncertainty. The final term is the **entropy of the final distribution** $p(x_T)$, representing the maximum noise.

The **main idea** is to decompose the entire process into how well the reverse process matches the true forward diffusion (**KL Divergence terms**) and the uncertainty introduced at each step (**entropy terms**).


### **Why does entropy simplify things?**

When we consider the entropy of the forward process, we are essentially saying: *"I already know how much noise I have introduced into the system"*. This value is a constant that can be calculated, which simplifies the expressions because we do not need to track every intermediate step of diffusion.
