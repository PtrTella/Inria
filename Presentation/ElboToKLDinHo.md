---
marp: true
math: mathjax
---

### **Starting Point (Variational Lower Bound)**

We begin from the variational lower bound (ELBO) that we derived in the previous section: 
$$
\mathcal{K} 

= \mathbb{E}_{q(x_{0:T})} \left[ \log \left( \frac{p_\theta(x_{0:T}) }{q(x_{1:T} | x_0)} \right) \right]

= \mathbb{E}_{q(x_{0:T})} \left[ 
  \log \left( \frac{p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} | x_t)}{\prod_{t=1}^T q(x_t | x_{t-1})} \right) 
\right]
$$

### **Expand the logarithm**
*Ho et al. (2020)* started the derivation by expanding the negative log-likelihood that for us is $\mathcal{-K}$ (instead of the log-likelihood):

$$
\mathcal{-K}
= \mathbb{E}_{q} \left[ -\log p(x_T) - \sum_{t=1}^T \log p_\theta(x_{t-1} | x_t) + \sum_{t=1}^T \log q(x_t | x_{t-1}) \right]
$$

---

### **Separate the final timestep**:
Split off the first transition $t = 1$ and isolate $\log p_\theta(x_0 | x_1)$:

$$
= \mathbb{E}_q \left[ 
    -\log p(x_T)
    - \sum_{t=2}^T \log p_\theta(x_{t-1} | x_t)
    - \log p_\theta(x_0 | x_1)
    + \sum_{t=2}^T \log q(x_t | x_{t-1})
    + \log q(x_1 | x_0)
\right]
$$

### **Group log-ratios**:
Combine log terms into ratios to express them more compactly:

$$
= \mathbb{E}_q \left[
    -\log p(x_T)
    - \sum_{t=2}^T \log \frac{p_\theta(x_{t-1} | x_t)}{q(x_t | x_{t-1})}
    - \log \frac{p_\theta(x_0 | x_1)}{q(x_1 | x_0)}
\right]
$$


---

### **Rearranging the terms**:

**Prior term:**

We expand $-\log p(x_T)$ using an algebraic trick:

$$
- \log p(x_T) = \log \frac{q(x_T | x_0)}{p(x_T)} \textcolor{blue}{- \log q(x_T | x_0)}
$$

This introduces a $\log q(x_T | x_0)$ term, which later cancels with another expression.

**Final term:**

We rewrite the last log-ratio:

$$
- \log \frac{p_\theta(x_0 | x_1)}{q(x_1 | x_0)} = \textcolor{green}{\log q(x_1 | x_0)} - \log p_\theta(x_0 | x_1)
$$

Here, $\log q(x_1 | x_0)$ **cancels** with the telescoping term under

---

**Intermediate terms:**
Bayes rule determine $q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}) \cdot q(x_{t-1} | x_0)}{q(x_t | x_0)}$. We rewrite the sum of transition ratios using log identity and substituting $q(x_{t} | x_{t-1})$ with the Bayes rule:

$$
- \sum_{t=2}^T \log \frac{p_\theta(x_{t-1} | x_t)}{q(x_t | x_{t-1})}
= \sum_{t=2}^T \log \Big( \frac{q(x_{t-1} | x_t, x_0)}{p_\theta(x_{t-1} | x_t)} \frac{q(x_t | x_0)}{q(x_{t-1} | x_0)} \Big)
$$

The last term is **telescopic** so we can rewrite it as:

$$
\sum_{t=2}^T \log \frac{q(x_t | x_0)}{q(x_{t-1} | x_0)}
= \log \frac{q(x_T | x_0)}{q(x_1 | x_0)}
= \textcolor{blue}{\log {q(x_T | x_0)}} \textcolor{green}{- \log {q(x_1 | x_0)}}
$$



---

### Final Result

After simplifications and cancellation of shared log-terms, we obtain:

$$
\mathbb{E}_{q(x_{0:T})} \left[
  \log \frac{q(x_T | x_0)}{p(x_T)}
  + \sum_{t=2}^T \log \frac{q(x_{t-1} | x_t, x_0)}{p_\theta(x_{t-1} | x_t)}
  - \log p_\theta(x_0 | x_1)
\right]
$$

This is Equation (21) — all terms now have interpretable structure (posterior over model prediction), and are ready to be expressed as KL divergences in the next step (Eq. 22).

---

### **Final KL Divergence Formulation**
$$
\mathcal{K} = \mathbb{E}_{q(x_{0:T})} \left[
  D_{KL}(q(x_T | x_0) || p(x_T))
  + \sum_{t=2}^T D_{KL}(q(x_{t-1} | x_t, x_0) || p_\theta(x_{t-1} | x_t))
  - \log p_\theta(x_0 | x_1)
\right]
$$

This is the final form of the KL divergence, which can be interpreted as a sum of KL divergences between the true forward process and the learned reverse process, plus a term for the initial state. This formulation is crucial for understanding how diffusion models learn to generate data by reversing the diffusion process.
