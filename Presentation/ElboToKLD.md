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

To **remove edge effects** at time $t = 0$, Sohl-Dickstein et al. set the final step of the reverse trajectory to be identical to the corresponding forward diffusion step:

$$
p_\theta(x_0 | x_1) = q(x_1 | x_0) \frac{\pi(x_0)}{\pi(x_1)}
$$

where $\pi(x_0)$ and $\pi(x_1)$ are reference distributions in the forward process. This assumption allows the cancellation of the initial term, so the summation in the integral starts from $t = 2$:

$$
K = \sum_{t=2}^T \int dx_{0:T} \, q(x_{0:T}) \log \left[\frac{p_\theta(x_{t-1} | x_t)}{q(x_t | x_{t-1})}\right] - H(p(x_T))
$$

---
## **Steps**

At the first time step $t = 1$, the term looks like:

$$
\mathbb{E}_{q(x_0, x_1)} \left[ \log \frac{p_\theta(x_0 | x_1)}{q(x_0 | x_1, x_0)} \right]
$$

We apply **Bayes' Rule** to express the reverse process:

$$
p_\theta(x_0 | x_1) = q(x_1 | x_0) \frac{\pi(x_0)}{\pi(x_1)}
$$

Now, we substitute this back into the KL term:

$$
\log \frac{p_\theta(x_0 | x_1)}{q(x_0 | x_1, x_0)} = \log \frac{q(x_1 | x_0) \frac{\pi(x_0)}{\pi(x_1)}}{q(x_0 | x_1, x_0)}
$$

---

Apply the properties of the logarithm to expand the expression:

$$
\log \frac{q(x_1 | x_0) \frac{\pi(x_0)}{\pi(x_1)}}{q(x_0 | x_1, x_0)} = \log q(x_1 | x_0) + \log \frac{\pi(x_0)}{\pi(x_1)} - \log q(x_0 | x_1, x_0)
$$

The key assumption in this derivation is that the two marginal distributions at the first time step are **equal** so $\pi(x_0) = \pi(x_1)$

This implies that the ratio simplifies:

$$
\frac{\pi(x_0)}{\pi(x_1)} = 1 \implies \log \frac{\pi(x_0)}{\pi(x_1)} = 0
$$

---

## **5️⃣ Total Simplification**

The integral at time $t = 1$ reduces to:

$$
\int q(x_0, x_1) \log \frac{q(x_1 | x_0)}{q(x_0 | x_1, x_0)} \, dx_0 dx_1
$$

Now, by the definition of conditional distributions in the forward process, $q(x_0 | x_1, x_0)$ is exactly matched by $q(x_1 | x_0)$, which means:

$$
q(x_0 | x_1, x_0) = q(x_0)
$$

Consequently, the integral becomes the logarithm of 1:

$$
\int q(x_0, x_1) \log 1 \, dx_0 dx_1 = 0
$$

---

## **6️⃣ Final Result**

This proves that the first term in the summation:

$$
\mathbb{E}_{q(x_0, x_1)} \left[ \log \frac{p_\theta(x_0 | x_1)}{q(x_0 | x_1, x_0)} \right]
$$

**completely cancels out to zero** due to the assumption that the two marginal distributions $\pi(x_0)$ and $\pi(x_1)$ are the same. This removes the edge effect at $t = 1$ from the total KL divergence.

---

## **7️⃣ Intuition Recap**

* The equality $\pi(x_0) = \pi(x_1)$ is crucial: it ensures the first KL term disappears.
* This means we can safely ignore the edge effect at the very first step of the reverse process.
* The model does not accumulate error at $t = 1$, making the reverse process smoother and more stable.

---

### 🚀 **Would you like me to extend this derivation to the next step ($t = 2$) to show how conditional entropies are introduced?**

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
K = \sum_{t=2}^T \int q(x_{0:T}) \log \left[ \frac{p_\theta(x_{t-1} | x_t)}{q(x_{t-1} | x_t, x_0)} \cdot \frac{q(x_{t-1} | x_0)}{q(x_t | x_0)} \right] dx_{0:T} - H(p(x_T))
$$

Now, the expression inside the logarithm is split into:

1. A **likelihood ratio**: $\frac{p_\theta(x_{t-1} | x_t)}{q(x_{t-1} | x_t, x_0)}$
2. A **prior ratio**: $\frac{q(x_{t-1} | x_0)}{q(x_t | x_0)}$

---

## **Introducing Conditional Entropies**

The paper then recognizes that several terms can be expressed as **conditional entropies**. Let's rewrite it explicitly:

$$
K = \sum_{t=2}^T \int q(x_{0:T}) \log \left[ \frac{p_\theta(x_{t-1} | x_t)}{q(x_{t-1} | x_t, x_0)} \right] dx_{0:T} + \sum_{t=2}^T \left[ H_q(X_t | X_0) - H_q(X_{t-1} | X_0) \right] - H(p(x_T))
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

The final expression becomes:

$$
K = -\sum_{t=2}^T \int dx_0 \, dx_t \, q(x_0, x_t) D_{KL}\left(q(x_{t-1} | x_t, x_0) || p_\theta(x_{t-1} | x_t)\right) + H(q(X_T | X_0)) - H(q(X_1 | X_0)) - H(p(X_T))
$$

---

## **Notes**
The **main term** is a **sum of KL divergences** for every step $t$ from 2 to $T$. The second and third terms are **conditional entropies** that represent the change in uncertainty. The final term is the **entropy of the final distribution** $p(x_T)$, representing the maximum noise.

The **main idea** is to decompose the entire process into how well the reverse process matches the true forward diffusion (**KL Divergence terms**) and the uncertainty introduced at each step (**entropy terms**).


### **Why does entropy simplify things?**

When we consider the entropy of the forward process, we are essentially saying: *"I already know how much noise I have introduced into the system"*. This value is a constant that can be calculated, which simplifies the expressions because we do not need to track every intermediate step of diffusion.
