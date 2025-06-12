import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from skimage import data, color
from skimage.transform import resize

# Step 1: Load and preprocess a grayscale image
image = color.rgb2gray(data.astronaut())
image = resize(image, (64, 64), anti_aliasing=True)
x0 = image.flatten()

# Step 2: Define the forward diffusion process parameters
T = 10  # number of timesteps
betas = np.linspace(0.0001, 0.02, T)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)

# Store distributions and degraded images
degraded_images = []
ratios = []

# Generate noisy images at each step
for t in range(T):
    mean = np.sqrt(alpha_bars[t]) * x0
    var = 1 - alpha_bars[t]
    xt = mean + np.random.normal(0, np.sqrt(var), size=x0.shape)
    degraded_images.append(xt)

# Step 3: Compute and store ratios between consecutive q(x_t|x0)
for t in range(1, T):
    mu1 = np.sqrt(alpha_bars[t]) * x0
    mu2 = np.sqrt(alpha_bars[t - 1]) * x0
    sigma1 = np.sqrt(1 - alpha_bars[t])
    sigma2 = np.sqrt(1 - alpha_bars[t - 1])
    
    # Compute ratio as Gaussian function
    ratio = norm.pdf(degraded_images[t], loc=mu1, scale=sigma1) / norm.pdf(degraded_images[t], loc=mu2, scale=sigma2)
    ratios.append(ratio)

# Step 4: Visualize

fig, axes = plt.subplots(2, T, figsize=(3 * T, 6))

# Plot degraded images
for t in range(T):
    axes[0, t].imshow(degraded_images[t].reshape(64, 64), cmap='gray')
    axes[0, t].set_title(f"x_{t}")
    axes[0, t].axis('off')

# Plot ratio heatmaps
for t in range(T - 1):
    axes[1, t].imshow(ratios[t].reshape(64, 64), cmap='gray')
    axes[1, t].set_title(f"ratio x_{t+1}/x_{t}")
    axes[1, t].axis('off')

axes[1, -1].axis('off')
plt.tight_layout()
plt.show()

# Step 5: Compute the Gaussian from the ratio for each pair of steps
mu_ratios = []
sigma_ratios = []

for t in range(1, T):
    alpha_bar_t = alpha_bars[t]
    alpha_bar_tm1 = alpha_bars[t - 1]

    sigma1_sq = 1 - alpha_bar_t
    sigma2_sq = 1 - alpha_bar_tm1

    inv_sigma_r_sq = (1 / sigma1_sq) - (1 / sigma2_sq)
    sigma_r_sq = 1 / inv_sigma_r_sq
    sigma_ratios.append(np.sqrt(sigma_r_sq))

    coef1 = np.sqrt(alpha_bar_t) / sigma1_sq
    coef2 = np.sqrt(alpha_bar_tm1) / sigma2_sq
    mu_r = sigma_r_sq * (coef1 - coef2) * x0
    mu_ratios.append(mu_r)

# Step 6: Visualize a comparison at a specific pixel (e.g., center pixel)
pixel_index = (64 * 32 + 32)  # center pixel

x_vals = np.linspace(-1.5, 1.5, 500)

fig, axes = plt.subplots(3, 3, figsize=(12, 9))
axes = axes.flatten()

for i in range(1, min(T, 10)):
    mu1 = np.sqrt(alpha_bars[i]) * x0[pixel_index]
    sigma1 = np.sqrt(1 - alpha_bars[i])
    
    mu2 = np.sqrt(alpha_bars[i - 1]) * x0[pixel_index]
    sigma2 = np.sqrt(1 - alpha_bars[i - 1])
    
    mu_r = mu_ratios[i - 1][pixel_index]
    sigma_r = sigma_ratios[i - 1]

    y1 = norm.pdf(x_vals, mu1, sigma1)
    y2 = norm.pdf(x_vals, mu2, sigma2)
    yr = norm.pdf(x_vals, mu_r, sigma_r)

    ax = axes[i - 1]
    ax.plot(x_vals, y1, label=f"q(x_{i}|x₀)", color='blue',linestyle='--')
    ax.plot(x_vals, y2, label=f"q(x_{i-1}|x₀)", color='green', linestyle='--')
    ax.plot(x_vals, yr, label="ratio ≈ N", color='red', linestyle='--')
    ax.set_title(f"Step {i}")
    ax.legend()

plt.tight_layout()
plt.show()
