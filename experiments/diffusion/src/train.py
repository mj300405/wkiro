import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from model import DiffusionModel
import torch.nn.functional as F

# Set device
device = (
    "mps" 
    if torch.backends.mps.is_available()
    else "cuda" 
    if torch.cuda.is_available() 
    else "cpu"
)
print(f"Using device: {device}")

class DiffusionTrainer:
    def __init__(self, 
                 model,
                 timesteps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 device=device):
        self.model = model.to(device)
        self.device = device
        self.timesteps = timesteps
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def get_noisy_image(self, x_start, t):
        noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def train_step(self, x, optimizer):
        optimizer.zero_grad()
        
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=self.device)
        x_noisy, noise = self.get_noisy_image(x, t)
        predicted_noise = self.model(x_noisy, t)
        
        loss = F.mse_loss(noise, predicted_noise)
        loss.backward()
        optimizer.step()
        
        return loss.item()

def train(model, 
          train_loader,
          num_epochs=100,
          lr=2e-4,
          save_dir='checkpoints',
          device=device):
    
    trainer = DiffusionTrainer(model, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            loss = trainer.train_step(images, optimizer)
            pbar.set_description(f"Epoch {epoch} | Loss: {loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))

def main():
    # Data transforms for MNIST
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to match model's expected input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Single channel normalization for MNIST
    ])
    
    # Load MNIST dataset from root data directory
    dataset = datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    
    # Initialize model with single channel input
    model = DiffusionModel(in_channels=1)
    
    # Train model
    train(model, train_loader)

if __name__ == '__main__':
    main() 