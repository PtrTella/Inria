---
marp: true
math: mathjax
---

# U-Net in Diffusion Models

---

## 💠 Overview: What is the U-Net in Diffusion Models?

In diffusion models, the core neural network is a **U-Net**. It's used to **predict the noise** $\epsilon$ that was added to a data sample $x_0$ during the forward noising process.

---

![](images/u-net-architecture.png)

---

## 💠 Input and Output

- **Input**: 
  - $x_t$ → a noisy version of a real sample (e.g., an image), with shape $(B, C, H, W)$,
  - $t$ → the timestep indicating how much noise has been added (a scalar per sample).
- **Output**:
  - $\epsilon_\theta(x_t, t)$ → a tensor of the same shape as $x_t$, representing the predicted noise.

---

## 💠 U-Net Architecture

#### 🔹 Encoder
Downsamples the input $x_t$ step-by-step, extracts increasingly abstract and compressed feature maps and Helps capture global structures.

#### 🔹 Decoder
Upsamples the compressed features back to the original spatial resolution, gradually reconstructs finer details, outputs a tensor predicting the noise $\epsilon$.

#### 🔹 Skip Connections
Direct links between encoder and decoder at matching resolutions, preserve high-frequency details lost during downsampling, crucial for maintaining local structure.

---
# 💠 Time and Text Embeddings

#### 🔹 How the timestep $t$ is used
Each $x_t$ corresponds to a different noise level depending on $t$.  
The network must be **aware** of which noise stage it is operating on.

The scalar $t$ is embedded into a high-dimensional vector $\text{Embed}(t)$ using sinusoidal positional encodings (as in Transformers) and is **Injected into each block** (e.g., ResNet blocks).

#### 🔹  Textual embedding

The **text** is injected into the **inner blocks** of the U-Net via a **Cross-Attention mechanism**.

This lets **every pixel** in the image decide **what parts of the prompt** are relevant, making the generation process **contextually guided** by the input text.

---
## 💠 Training steps

1. Sample clean data: $x_0$,
2. Sample timestep: $t \sim \text{Uniform}(1, T)$,
3. Sample noise: $\epsilon \sim \mathcal{N}(0,I)$,
4. Generate noised sample:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon
$$

5. Predict noise:

$$
\hat{\epsilon} = \epsilon_\theta(x_t, t)
$$

6. Minimize loss:

$$
\mathcal{L}_{\text{simple}} = \|\epsilon - \hat{\epsilon}\|^2
$$

The network learns to predict the added noise at every level of degradation.

---


## 💠 ResNet Block

A **ResNet Block** receives an input feature map $h$ (e.g., $h \in \mathbb{R}^{B \times C \times H \times W}$) and performs:

1. A first 2D convolution ($\text{Conv1}$),
2. A non-linear activation (ReLu, SiLU),
3. A second 2D convolution ($\text{Conv2}$),
4. **Adds** the input back at the end: $\text{output} = h + F(h)$

### 🔹 Timestep Injection

When a timestep embedding $t$ is present usaully small MLP (neural network) computes $\gamma(t)$ and $\beta(t)$, these are used to **modulate** the feature map and concatenated into Resnet Block.

$$
h' = \gamma(t) \cdot h + \beta(t)
$$

---

## 💠 Downsampling (Spatial Compression)

**Downsampling** reduces the spatial resolution while increasing or preserving the number of channels. Helps capture **global features**.

Obtained with **MaxPooling** (selects maximum value in local windows) or **Convolution with stride 2** ( that skips pixels during convolution). 

Mathematically (convolution stride 2):

$$
y[i, j] = \sum_{m,n} W[m,n] \cdot x[2i+m, 2j+n]
$$

$x$	Input feature map (higher resolution), the original image
$y$	Output feature map (lower resolution), after downsampling
$W[m, n]$	Convolutional kernel (or filter), a small matrix of learnable weights

---

## 💠 Upsampling (Spatial Expansion)

Carried out with **Nearest neighbor upsampling** (simple pixel duplication) and **Transposed Convolution** (learned upsampling).

Mathematically (transposed convolution):

$$
y[i,j] = \sum_{m,n} W[m,n] \cdot x[\lfloor i/2 \rfloor + m, \lfloor j/2 \rfloor + n]
$$

$x$ Input feature map (lower resolution), the small image
$y$ Output feature map (higher resolution), the upsampled image
$W$ Convolutional kernel (filter), a small matrix of learnable weights

---

## 💠 3. Cross-Attention Block

A Cross-Attention Block allows the model to incorporate information from the text prompt during the denoising process in a DDPM (Denoising Diffusion Probabilistic Model).

The image feature map (from the U-Net) provides the queries $Q$.

The text embedding (from a text encoder like CLIP or T5) provides the keys $K$ and values $V$.

The model learns how much each spatial location in the image should focus on each word in the prompt.

---

Given projections:

- $Q = W_Q h$ (from image features),
- $K = W_K c$ (from text embedding),
- $V = W_V c$ (again from text embedding),

the Cross-Attention is computed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where $d_k$ is the key dimension.

---

#  U-Net Python Keras Implementation
```
# CONVOLUTIONAL BLOCK: two Conv with ReLu activation
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = Activation("relu")(x)
    # RESNET BLOCK: Add time embedding
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = Activation("relu")(x)
    return x

# ENCODER BLOCK: Conv block followed by maxpooling
def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p   

# DECODER BLOCK: skip features gets input from encoder for concatenation
def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x
```
---

```
# U-Net BUILD
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge
    # ATTENTION BLOCK for textual embedding

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(3, 1, padding="same")(d4) # Same shape of original image (supposed rgb)

    model = Model(inputs, outputs, name="U-Net")
    return model
```







