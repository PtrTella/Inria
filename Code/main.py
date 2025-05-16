import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from DiffusionModel import DiffusionModel, NoisePredictor
from DiffusionVisualizer import generate_and_visualize
import os

# Configurazione
IMG_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
TIMESTEPS = 1000
CHECKPOINT_PATH = "./checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Trasformazioni delle immagini
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset e DataLoader
dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Inizializzazione del modello
diffusion_model = DiffusionModel(img_size=IMG_SIZE, timesteps=TIMESTEPS)
noise_predictor = NoisePredictor(img_size=IMG_SIZE, channels=CHANNELS)

# Controllo di checkpoint esistente
checkpoint_file = os.path.join(CHECKPOINT_PATH, "noise_predictor.pth")
if os.path.exists(checkpoint_file):
    print("Checkpoint trovato, caricamento del modello...")
    noise_predictor.load_state_dict(torch.load(checkpoint_file))
    print("Modello caricato correttamente!")
else:
    print("Checkpoint non trovato, si procederà con l'addestramento.")

# Addestramento del modello
print("Inizio addestramento...")
diffusion_model.train_model(noise_predictor, dataloader, epochs=EPOCHS, lr=LEARNING_RATE)
print("Addestramento completato!")

# Salvataggio del modello
torch.save(noise_predictor.state_dict(), checkpoint_file)
print(f"Modello salvato in: {checkpoint_file}")

# Esempio di generazione e visualizzazione
prompt = "A beautiful sunset over the mountains"
generate_and_visualize(diffusion_model, noise_predictor, prompt, (1, CHANNELS, IMG_SIZE, IMG_SIZE), TIMESTEPS)
