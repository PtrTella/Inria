---
marp: true
---


# 📚 Complete Story: From Likelihood to MSE Loss in Diffusion Models

---

# 🎯 Goal of training

The goal in training diffusion models is to **maximize the likelihood** of real data samples $x_0$:

$$
\log p_\theta(x_0)
$$

where:
- $p_\theta$ is the **model distribution**,
- $\theta$ are the **parameters** of the neural network.

✅ In plain words:  
We want our model to assign **high probability** to real data.

---

Computing the integral is **intractable** because $p_\theta(x_0)$ is **marginalized** over many latent variables:

$$
p_\theta(x_0) = \int p_\theta(x_{0:T}) \, dx_{1:T}
$$

where:
- $x_1, ..., x_T$ are intermediate **noised versions** of the data.
- $p_\theta(x_{0:T})$ is the **joint reverse process**:
  $$
  p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} | x_t)
  $$
  with:
  - $p(x_T)$ = simple Gaussian prior (pure noise),
  - $p_\theta(x_{t-1}|x_t)$ = learned reverse transitions.

---

# 🎯 Introduce a variational distribution $q(x_{1:T}|x_0)$

We introduce a **known** forward process $q$, which is a **fixed**, **predefined** noising process, and is simple to sample from and compute. At each step, we add small Gaussian noise.

### Forward process:

$$
q(x_{1:T} | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})
$$

where:

$$
q(x_t|x_{t-1}) = \mathcal{N}\left( \sqrt{1-\beta_t} x_{t-1}, \beta_t I \right)
$$



---

# 🎯 Apply Variational Inference (ELBO)

We write:

$$
\log p_\theta(x_0) = \log \int \textcolor{blue}{q(x_{1:T}|x_0)} \frac{p_\theta(x_{0:T})}{\textcolor{blue}{q(x_{1:T}|x_0)} } dx_{1:T} = \log \mathbb{E}_{q(x_{1:T}|x_0)}\left[ \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \right]
$$

Applying **Jensen's inequality**:

$$
\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T}|x_0)} \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \right]
$$
Define it as the **Evidences Lower Bound**
$$
\mathcal{L}_{\text{vlb}} = \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log p_\theta(x_{0:T}) - \log q(x_{1:T}|x_0)\right]
$$

---
# 🎯 Expand **$p_\theta$** and **$q$** defined as forward and reverse process and Rearranging terms into KL Divergence


$$
\mathcal{L}_{\text{vlb}} = \mathbb{E}_q \left[ \log p(x_T) + \sum_{t=1}^T \log p_\theta(x_{t-1}|x_t) - \sum_{t=1}^T \log q(x_t|x_{t-1}) \right]
$$


The paper rearrange this expression by separate the terms involving $t=T$ (for $x_T$) and write **KL divergences** between the "true" forward conditionals and the learned reverse conditionals.

$$
\mathcal{L}_{\text{vlb}} = \mathbb{E}_q \left[
D_{\text{KL}}\left(q(x_T|x_0) \,\|\, p(x_T)\right)
+ \sum_{t=2}^T D_{\text{KL}}\left(q(x_{t-1}|x_t,x_0) \,\|\, p_\theta(x_{t-1}|x_t)\right)
- \log p_\theta(x_0|x_1)
\right]
$$

---

# 🧠 Step 4: Now: **Why are some terms ignored or treated differently?**

### 🔥 First Term — $D_{\text{KL}}(q(x_T|x_0) \| p(x_T))$

- In practice, since the **forward process is very long** (with many small noise additions $\beta_t$),
- At the end, $q(x_T|x_0)$ becomes **almost identical** to $p(x_T)$,
- This KL divergence becomes **very small** and can be **ignored** (or computed once and treated as constant).

✅ **Practical result**: often neglected or considered constant.

---

### 🔥 Third Term $-\log p_\theta(x_0|x_1)$

- This term is more complicated because it would require accurately estimating $p_\theta(x_0|x_1)$,
- But it is often approximated or **replaced** during training,
- In "simple loss" variants, it is typically **ignored** or handled in an alternative way.

✅ **Practical result**: either approximated or replaced by directly predicting the noise ($\epsilon$).

---

Would you also like me to make a **full bilingual version** (Italian-English side-by-side) to help you check your own notes? 🚀  
Let’s keep going if you want! 🎯
---

# 🧠 Now: What are each of these terms?

| Term | Mathematical Expression | Meaning |
|:---|:---|:---|
| **Prior KL** | $D_{\text{KL}}(q(x_T|x_0) \,\|\, p(x_T))$ | Makes sure that final noise $x_T$ is close to standard Gaussian noise |
| **Transition KLs** | $\sum_{t=2}^T D_{\text{KL}}(q(x_{t-1}|x_t,x_0) \,\|\, p_\theta(x_{t-1}|x_t))$ | Makes sure that each reverse step is close to the "true" reverse given by forward process |
| **Reconstruction term** | $-\log p_\theta(x_0|x_1)$ | Makes sure that $x_1$ can reconstruct $x_0$ correctly |

---

# 🎯 Step 7: Focus on Transition KLs

The main loss component during training is:

$$
\sum_{t=2}^T D_{\text{KL}}\left(q(x_{t-1}|x_t,x_0) \,\|\, p_\theta(x_{t-1}|x_t)\right)
$$

What are we comparing?

| Object | Definition | Explanation |
|:---|:---|:---|
| **$q(x_{t-1}|x_t,x_0)$** | True posterior computed from the forward process | Can be **explicitly calculated** using Gaussian rules |
| **$p_\theta(x_{t-1}|x_t)$** | Neural network model prediction | What the model tries to learn |

✅ **Important**: Both $q$ and $p_\theta$ are Gaussian distributions.

---

# 🎯 Step 8: KL divergence between two Gaussians

When both distributions are Gaussians:

- If variances are **fixed** (e.g., set to $\beta_t I$),
- Then the KL divergence reduces to:

$$
D_{\text{KL}} \propto \| \text{true mean} - \text{predicted mean} \|^2
$$

✅ **Thus:**  
**Minimizing the KL divergence becomes minimizing a Mean Squared Error (MSE) between means**.

---

# 🎯 Step 9: Predicting the Noise $\epsilon$

Instead of predicting directly the means,  
the paper **Ho et al. (2020)** notices that it is **even easier to predict the noise** $\epsilon$.

From the forward process:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
\quad , \quad \epsilon \sim \mathcal{N}(0, I)
$$

Thus:

- If we know $x_t$,
- The **noise $\epsilon$** that was added is fully determined.

✅ **Idea**:  
Train the model $\epsilon_\theta(x_t,t)$ to predict the added noise!

---

# 🎯 Step 10: Final Practical Loss (Simple MSE Loss)

Thus, the final loss becomes:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

✅ In plain words:

- Sample a real data point $x_0$,
- Sample a timestep $t$,
- Sample noise $\epsilon$,
- Compute $x_t$,
- Predict $\epsilon$ from $x_t$,
- Minimize the squared error between the true and predicted $\epsilon$.

---

# 🧠 Visual Global Summary

| Stage | Action |
|:---|:---|
| Define forward process $q(x_{1:T}|x_0)$ | Add noise step-by-step |
| Write variational bound (ELBO) | Lower bound for $\log p_\theta(x_0)$ |
| Expand ELBO into KL terms | Measure how close model is to true reverse |
| KL divergence between Gaussians → MSE | Simplification |
| Predict noise $\epsilon$ instead of mean | Easier, more stable |
| Final training loss: simple MSE on $\epsilon$ | Standard supervised training |

---

# 🚀 Summary in one diagram

```
[x0] --forward q--> [x1] --q--> [x2] --q--> ... --q--> [xT] (pure noise)

During training:
- sample t
- sample noise ε
- create x_t = sqrt(ᾱ_t)x_0 + sqrt(1-ᾱ_t)ε
- predict ε_θ(x_t, t)
- minimize ||ε - ε_θ(x_t, t)||²
```

---

# 📣 Would you like now:

- To **see a concrete numerical example** (with actual numbers, like $x_0 = 1.0$, $\beta = 0.01$)?
- Or **a PyTorch mini-code** implementing exactly these steps?

🚀  
I strongly recommend doing both — it would **lock everything firmly** in your mind!  
Should we continue? 🎯✨