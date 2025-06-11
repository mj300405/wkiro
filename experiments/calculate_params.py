import torch
from gan.src.model import Generator, Discriminator
from vq_vae.src.model import VQVAE
from basic_ae.src.model import BasicAE
from cond_vae.src.model import ConditionalVAE
from conv_vae.src.model import ConvVAE
from diffusion.src.model import DiffusionModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_params(model_name, model):
    total_params = count_parameters(model)
    print(f"\n{model_name}:")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Size in MB: {total_params * 4 / (1024 * 1024):.2f}")

# Initialize models
models = {
    "GAN Generator": Generator(),
    "GAN Discriminator": Discriminator(),
    "VQ-VAE": VQVAE(),
    "Basic AE": BasicAE(),
    "Conditional VAE": ConditionalVAE(),
    "Convolutional VAE": ConvVAE(),
    "Diffusion Model": DiffusionModel()
}

# Calculate and print parameters for each model
for name, model in models.items():
    print_model_params(name, model) 