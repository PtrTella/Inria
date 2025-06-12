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
T = 50  # number of timesteps
betas = np.linspace(0.0001, 0.02, T)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)

# Step 3: Generate degraded images at each step
degraded_images = []
for t in range(T):
    mean = np.sqrt(alpha_bars[t]) * x0
    var = 1 - alpha_bars[t]
    xt = mean + np.random.normal(0, np.sqrt(var), size=x0.shape)
    degraded_images.append(xt)

# Step Compute and store ratios between consecutive q(x_t|x0)
ratios = []
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

# Step 4: Compute Gaussian ratio parameters using reversed ratio (q(x_{t-1}|x0)/q(x_t|x0))
mu_ratios = []
sigma_ratios = []
valid_steps = []

for t in range(1, T):
    alpha_bar_t = alpha_bars[t]
    alpha_bar_tm1 = alpha_bars[t - 1]

    sigma_t_sq = 1 - alpha_bar_t
    sigma_tm1_sq = 1 - alpha_bar_tm1

    inv_sigma_r_sq = (1 / sigma_tm1_sq) - (1 / sigma_t_sq)
    if inv_sigma_r_sq <= 0:
        continue  # skip invalid ratio

    sigma_r_sq = 1 / inv_sigma_r_sq
    sigma_r = np.full_like(x0, np.sqrt(sigma_r_sq))

    coef_tm1 = np.sqrt(alpha_bar_tm1) / sigma_tm1_sq
    coef_t = np.sqrt(alpha_bar_t) / sigma_t_sq
    mu_r = sigma_r_sq * (coef_tm1 - coef_t) * x0

    sigma_ratios.append(sigma_r)
    mu_ratios.append(mu_r)
    valid_steps.append(t)

# Step 5: Visualize mu and sigma ratio over entire image
if valid_steps:

    for i, t in enumerate(valid_steps):
        print(mu_ratios[i])
        print(sigma_ratios[i].reshape(64, 64))

else:
    print("Nessun passo valido per calcolare e visualizzare le gaussiane ratio invertite sull'immagine.")


# Selezioniamo alcuni step validi per il grafico delle curve
import matplotlib.pyplot as plt

x_vals = np.linspace(-1.5, 1.5, 500)

# Troviamo i primi 6 step validi (o meno)
num_plots = min(6, len(valid_steps))
fig, axes = plt.subplots(2, num_plots, figsize=(5 * num_plots, 8))

for i in range(num_plots):
    t = valid_steps[i]
    
    # Calcoliamo mu e sigma per i tre gaussiani (q_t-1, q_t, ratio)
    mu_tm1 = np.sqrt(alpha_bars[t - 1]) * x0
    sigma_tm1 = np.sqrt(1 - alpha_bars[t - 1])
    
    mu_t = np.sqrt(alpha_bars[t]) * x0
    sigma_t = np.sqrt(1 - alpha_bars[t])
    
    mu_r = mu_ratios[i]
    sigma_r = sigma_ratios[i]
    
    # Selezioniamo un campione di 200 pixel distribuiti
    sampled_idx = np.linspace(0, len(x0) - 1, 400, dtype=int)
    
    # Calcoliamo e media le curve PDF su questi pixel
    y_tm1 = np.mean([norm.pdf(x_vals, mu_tm1[j], sigma_tm1) for j in sampled_idx], axis=0)
    y_t = np.mean([norm.pdf(x_vals, mu_t[j], sigma_t) for j in sampled_idx], axis=0)
    y_r = np.mean([norm.pdf(x_vals, mu_r[j], sigma_r[j]) for j in sampled_idx], axis=0)

    # Plottiamo le curve nel subplot
    ax = axes[0, i]
    ax.plot(x_vals, y_tm1, label=f"q(x_{t-1}|x₀)", color='green', linestyle='--')
    ax.plot(x_vals, y_t, label=f"q(x_{t}|x₀)", color='blue', linestyle='--')
    ax.plot(x_vals, y_r, label="ratio ≈ N", color='red')
    ax.set_title(f"Step {t}")
    ax.legend()

    # Plottiamo la differenza (y_tm1 / y_t) vs y_r per confronto
    ax2 = axes[1, i]
    true_ratio = y_tm1 / y_t
    ax2.plot(x_vals, true_ratio, label="q(t-1)/q(t) (true)", color='black')
    ax2.plot(x_vals, y_r, label="ratio ≈ N", color='red', linestyle='--')
    ax2.set_title(f"Step {t} - Confronto ratio")
    ax2.legend()

plt.tight_layout()
plt.show()


# Reimplementiamo la ricostruzione backward partendo da x_T e usando le gaussiane ratio
# La ricostruzione sarà fatta via "product of conditionals" partendo da x_T

# Uniamo la ricostruzione deterministica e stocastica con confronto visivo

# Inizializzazione
xT = degraded_images[-1]
xt_reconstructed = xT.copy()
xt_stochastic = xT.copy()

reconstructed_steps = [xt_reconstructed.reshape(64, 64)]
stochastic_steps = [xt_stochastic.reshape(64, 64)]

# Applichiamo entrambe le ricostruzioni (deterministica e stocastica)
for i in reversed(range(len(valid_steps))):
    mu_r = mu_ratios[i]
    sigma_r = sigma_ratios[i]

    # Ricostruzione deterministica
    xt_reconstructed = mu_r + (xt_reconstructed - mu_r) * sigma_r
    reconstructed_steps.append(xt_reconstructed.reshape(64, 64))

    # Ricostruzione stocastica
    eps = np.random.normal(0, 1, size=mu_r.shape)
    xt_prev_stoch = mu_r + sigma_r * eps
    stochastic_steps.append(xt_prev_stoch.reshape(64, 64))
    xt_stochastic = xt_prev_stoch

# Visualizzazione: 3 righe (rec deterministica, rec stocastica, degraded)
fig, axes = plt.subplots(3, len(reconstructed_steps), figsize=(3 * len(reconstructed_steps), 9))

for i in range(len(reconstructed_steps)):
    axes[0, i].imshow(reconstructed_steps[i], cmap='gray')
    axes[0, i].set_title(f"rec det {i}")
    axes[0, i].axis('off')

    axes[1, i].imshow(stochastic_steps[i], cmap='gray')
    axes[1, i].set_title(f"rec stoch {i}")
    axes[1, i].axis('off')

    axes[2, i].imshow(degraded_images[T - i - 1].reshape(64, 64), cmap='gray')
    axes[2, i].set_title(f"q(x{T - i - 1})")
    axes[2, i].axis('off')

plt.tight_layout()
plt.show()
