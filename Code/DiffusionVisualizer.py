import matplotlib.pyplot as plt


def visualize_diffusion(images, steps=10):
    """
    Visualizza i passaggi di diffusione inversa.
    - images: lista di immagini generate durante il sampling
    - steps: numero di step da visualizzare (equidistanti)
    """
    num_images = len(images)
    indices = list(range(0, num_images, max(1, num_images // steps)))

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        img = images[idx].detach().cpu().numpy().transpose(1, 2, 0)
        plt.subplot(1, len(indices), i + 1)
        plt.imshow((img + 1) / 2)  # Normalize to [0, 1]
        plt.axis('off')
        plt.title(f"Step {idx}")
    plt.show()

def generate_and_visualize(model, noise_predictor, prompt, img_shape, num_steps):
    """
    Genera immagini e visualizza il processo di diffusione inversa.
    """
    print(f"Generazione immagine per il prompt: '{prompt}'")
    images = model.sample_images(noise_predictor, img_shape, num_steps, prompt)
    visualize_diffusion(images)
