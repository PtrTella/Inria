---
marp: true
math: mathjax
---

## Analytical Evaluation of the Proposed Caching Architecture

---

### Caching Scope: ResNet-18 Substructure

> "We distilled the first three layers of ResNet-18... These three layers of ResNet-18 are the ‘teacher’ model in our distillation process." \[p. 9]

The provided architectural (Appendix 7.1–7.3) clarify that this actually refers to **the first three ResNet blocks**. This includes:

* The initial convolution + batch norm + ReLU + MaxPooling
* **Layer1**: 2 × BasicBlock(64 → 64)
* **Layer2**: 2 × BasicBlock(64 → 128)
* **Layer3**: 2 × BasicBlock(128 → 256)

The final output shape of this substructure is `[32, 256, 14, 14]`.

---

### The Role of the Architectures: SmallModel vs Replaced Substructure

In the proposed design, two architectures serve distinct roles:

* **Replaced Substructure (Ns)**: This is a full replica of the first three ResNet blocks, serving as the "teacher" for both training and fallback inference. If memoization fails, this network is executed to produce the necessary intermediate activations.

* **SmallModel (Ds)**: This is a lightweight version of `Ns`, with approximately **half the number of residual blocks per stage**, trained to mimic the output of `Ns`. Its purpose is to produce a low-cost estimate that is used to **match against cached outputs**.

---

### The Caching Mechanism: Channel-Level Memoization
The pipeline works as follows:

1. Given input $x$, compute $Ds(x)$
2. Search in a lookup table for a cached output $y^*$ close to $Ds(x)$
3. If the match is within a threshold $T$, reuse $y^*$; otherwise, compute $Ns(x)$ and cache it

This process is done at the **channel level**, not at the full feature map level.

--- 

### The Memoization Limitation: Channel-Wise Cache ≠ Full Skipping

As stated in the paper:

> “Even though we successfully avoided the computation of the first three layers for 20% of the caches, the remaining 80% still required their full computation” \[p. 3]

This means that only 20% of the output channels from the final substructure layer (e.g., 51 out of 256) can be skipped via caching. The other 80% must still be computed via `Ns`.

---

### Analytical Modeling of Computation Cost

Let’s define:
* $L$: the number of layers in the substructure $N_s$ (in this case: 3)
* $C_{l}$: the total number of output channels from $N_s$ (depending on layer)
* $c$: the number of channels actually memoized (20% of $C_{l}$)
* $f(l, c)$: the computational cost (in FLOPs) of layer $l$ for $c$ output channels
* $R_{\text{Ns}}$: computational cost of the full substructure
* $R_{\text{Ds}}$: cost of the student model
* $R_{\text{eff}}$: effective runtime cost with caching


#### Total ResNet Cost without Caching (Baseline):

$$
R_{\text{Ns}} = \sum_{l=1}^{L} f(l, C_{l})
$$

---
 
#### Cost with Caching (Optimistidc Propose):
In this propose we assume that we can idelally compute, for all layers, only the channels that are not cached, which is $C_l - c$ (pecentage of $C_l$):

$$
R_{\text{eff}} = R_{\text{Ds}} + \sum_{l=1}^{L} f(l, C_{l} - c)
$$

Where:

* $R_{\text{Ds}}$ is the cost of the student model, which is always evaluated for every input. It is 50% lighter than the original substructure (paper architecture).
#### Net Gain (Target):

$$
\Delta R = R_{\text{Ns}} - R_{\text{eff}} = \underbrace{\sum_{l=1}^{L} f(l, C_{l})}_{\text{baseline}} - \left( R_{\text{Ds}} + \sum_{l=1}^{L} f(l, C_{l} - c) \right)
$$

---

#### Estimating $f(l, c)$:

For convolutional layers, the number of floating point operations (FLOPs) can be approximated as:

$$
f(l, C_{out}) = 2 \cdot H_l \cdot W_l \cdot K^2 \cdot C_{in} \cdot C_{out} \textrm{ (FLOPs)}
$$

Where:

* $H_l$, $W_l$: spatial dimensions of the output of layer $l$
* $K$: kernel size (typically 3)
* $C_{in}$ number of input channels to layer $l$
* $C_{out}$: number of output channels (either $C_l$ or $C_l - c$)

The factor 2 accounts for both the multiplication and addition per MAC operation.

---

### Numerical Estimate Based on FLOPs

Using architecture parameters extracted from the document (Appendix 7.3), the total FLOPs required to compute `Ns` were estimated as:

$$
R_{\text{Ns}} \approx 2.78 \times 10^9 \text{ FLOPs}
$$

Assuming:

* The student model is 50% lighter: $R_{\text{Ds}} \approx 1.39 \times 10^9$
* Only 20% of the channels are cached so $C_l-c = 0.8C_l$

We calculate:

$$
R_{\text{eff}} = R_{\text{Ds}} +  R_{\text{Ns}} = 1.39 \times 10^9 + 1.78 \times 10^9 = 3.17 \times 10^9
$$

This is **greater than the original** $R_{\text{Ns}}$, meaning the caching system currently increases computation time by \~14%.

---

### Final Considerations

The author’s method demonstrates that channel-level memoization is **functionally feasible** and can preserve partially accuracy. However, the **current architectural design does not allow full skipping of the `Ns` substructure unless 100% of the channels are matched in the cache**.

This reveals the importance of identifying **substructures that can be entirely skipped** or reformulating the caching granularity to operate on **entire blocks**, not channels.

---
### References

[Chenna, Dwith. "Evolution of Convolutional Neural Network (CNN): Compute vs Memory bandwidth for Edge AI." arXiv preprint arXiv:2311.12816 (2023).](https://arxiv.org/abs/2311.12816)

[Scirp.org](https://www.scirp.org/journal/paperinformation?paperid=94510&utm_source=chatgpt.com) -- In contrast, batch normalization layers come with negligible computational cost, but significant data transfer time
[Epoch.ai](https://epoch.ai/blog/backward-forward-FLOP-ratio?utm_source=chatgpt.com) -- Flops considerations