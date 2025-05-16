import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from transformers import CLIPTextModel, CLIPTokenizer

class DiffusionModel:
    def __init__(self, img_size, timesteps, beta_start=1e-4, beta_end=0.02):
        """
        Inizializzazione del modello di diffusione:
        - img_size: dimensione dell'immagine (assumiamo quadrata per semplicità)
        - timesteps: numero di step temporali di diffusione
        - beta_start, beta_end: valori iniziale e finale del rumore
        """
        self.img_size = img_size
        self.timesteps = timesteps
        
        # Definizione del noise schedule
        self.beta = np.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = np.cumprod(self.alpha)

        # Text Encoder (CLIP)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    def get_text_embedding(self, prompt):
        """
        Ottiene l'embedding testuale dato un prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            text_features = self.text_encoder(**inputs).last_hidden_state
        return text_features

    def forward_diffusion(self, x0, t):
        """
        Applica il processo di diffusione in avanti:
        q(x_t | x_0) = N(x_t; sqrt(alpha_hat_t) * x0, (1 - alpha_hat_t) * I)
        
        x0: batch di immagini originali
        t: step temporale di diffusione
        """
        noise = torch.randn_like(x0)
        sqrt_alpha_hat = np.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = np.sqrt(1 - self.alpha_hat[t])
        xt = sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise
        return xt, noise
    
    def sample_from_forward(self, x0, num_steps):
        """
        Campionamento dal processo di diffusione in avanti:
        - x0: immagine di partenza
        - num_steps: numero di step di campionamento
        """
        images = []
        for t in range(num_steps):
            xt, _ = self.forward_diffusion(x0, t)
            images.append(xt)
        return images

    def reverse_diffusion(self, xt, t, noise_pred):
        """
        Applica il processo di diffusione inversa:
        p(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(xt, t), beta_t I)

        xt: immagine noised al tempo t
        t: step temporale
        noise_pred: predizione del rumore dalla rete neurale
        """
        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_hat_t = self.alpha_hat[t]
        
        mu_theta = (1 / np.sqrt(alpha_t)) * (xt - (beta_t / np.sqrt(1 - alpha_hat_t)) * noise_pred)
        sigma_t = np.sqrt(beta_t)

        # Sampling inverso
        noise = torch.randn_like(xt)
        xt_prev = mu_theta + sigma_t * noise
        return xt_prev
    
    def train_model(self, model, dataloader, epochs=10, lr=1e-4):
        """
        Loop di addestramento del modello di diffusione:
        - model: rete neurale per predire il rumore
        - dataloader: caricamento del dataset
        - epochs: numero di epoche di addestramento
        - lr: learning rate
        """
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            for images, captions in dataloader:
                t = torch.randint(0, self.timesteps, (1,)).item()
                xt, noise = self.forward_diffusion(images, t)
                
                # Ottengo l'embedding del testo
                text_emb = self.get_text_embedding(captions[0])
                
                noise_pred = model(xt, t, text_emb)
                
                loss = criterion(noise_pred, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f"Epoca {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def sample_images(self, model, img_shape, num_steps, prompt):
        """
        Genera immagini a partire dal puro rumore usando il processo inverso.
        - model: modello di predizione del rumore
        - img_shape: dimensione dell'immagine da generare
        - num_steps: numero di passi di sampling
        - prompt: testo di condizionamento
        """
        x_t = torch.randn(img_shape)  # Partenza dal rumore puro
        text_emb = self.get_text_embedding(prompt)
        images = [x_t]
        
        for t in reversed(range(num_steps)):
            noise_pred = model(x_t, t, text_emb)
            x_t = self.reverse_diffusion(x_t, t, noise_pred)
            images.append(x_t)
        
        return images

class NoisePredictor(nn.Module):
    def __init__(self, img_size, channels=3, text_dim=512):
        """
        Rete neurale per predire il rumore ε_θ(xt, t), condizionata sul testo.
        """
        super(NoisePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 3, stride=2, padding=1, output_padding=1)
        )
        self.text_embedding = nn.Linear(text_dim, img_size * img_size)

    def forward(self, xt, t, text_emb):
        """
        Predizione del rumore dato uno stato xt, il tempo t e l'embedding testuale.
        """
        text_cond = self.text_embedding(text_emb).view(-1, 1, xt.shape[2], xt.shape[3])
        xt = xt + text_cond
        return self.net(xt)
