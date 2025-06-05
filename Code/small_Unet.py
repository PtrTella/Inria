import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pathlib import Path

from small_unet import SmallUNet
from simple_diffusion import GaussianDiffusion, logsnr_schedule_cosine

# --- Configurazione ---
image_size = 32
channels = 3
batch_size = 64
epochs = 100
lr = 1e-4
save_dir = Path("./results_ddpm")
save_dir.mkdir(exist_ok=True, parents=True)

# --- Dataset CIFAR-10 ---
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# --- Modello ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallUNet(in_channels=3, out_channels=3, dual_output=False).to(device)
ddpm = GaussianDiffusion(
    model,
    image_size=image_size,
    channels=channels,
    pred_objective='eps',
    noise_schedule=logsnr_schedule_cosine,
    num_sample_steps=500
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

# --- Training Loop ---
for epoch in range(epochs):
    model.train()
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)

        loss = ddpm(imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"[Epoch {epoch} Batch {i}] Loss: {loss.item():.4f}")

    # Sample immagini
    model.eval()
    with torch.no_grad():
        samples = ddpm.sample(batch_size=16)
        save_image(samples, save_dir / f"sample_epoch_{epoch}.png", nrow=4)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pathlib import Path

from small_unet import SmallUNet
from simple_diffusion import GaussianDiffusion, logsnr_schedule_cosine

# --- Configurazione ---
image_size = 32
channels = 3
batch_size = 64
epochs = 100
lr = 1e-4
save_dir = Path("./results_ddpm")
save_dir.mkdir(exist_ok=True, parents=True)

# --- Dataset CIFAR-10 ---
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# --- Modello ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallUNet(in_channels=3, out_channels=3, dual_output=False).to(device)
ddpm = GaussianDiffusion(
    model,
    image_size=image_size,
    channels=channels,
    pred_objective='eps',
    noise_schedule=logsnr_schedule_cosine,
    num_sample_steps=500
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

# --- Training Loop ---
for epoch in range(epochs):
    model.train()
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)

        loss = ddpm(imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"[Epoch {epoch} Batch {i}] Loss: {loss.item():.4f}")

    # Sample immagini
    model.eval()
    with torch.no_grad():
        samples = ddpm.sample(batch_size=16)
        save_image(samples, save_dir / f"sample_epoch_{epoch}.png", nrow=4)
