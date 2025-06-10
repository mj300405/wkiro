import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image
import os
import numpy as np
from tqdm import tqdm
from model import DiffusionModel
from datetime import datetime

device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

def main():
    # Data
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    # Model
    model = DiffusionModel(in_channels=1, base_ch=32).to(device)
    # DDPM schedule
    timesteps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    # Output dir
    run_id = datetime.now().strftime('run_%Y%m%d_%H%M%S')
    run_dir = os.path.join('checkpoints', run_id)
    os.makedirs(run_dir, exist_ok=True)
    # Training
    for epoch in range(1, 51):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        losses = []
        for x, _ in pbar:
            x = x.to(device)
            t = torch.randint(0, timesteps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
            x_noisy = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
            pred_noise = model(x_noisy, t)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch} | Loss: {np.mean(losses):.4f}")
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(run_dir, f'checkpoint_epoch_{epoch}.pt'))
            # Sampling
            model.eval()
            with torch.no_grad():
                x = torch.randn(16, 1, 32, 32).to(device)
                for t_ in reversed(range(timesteps)):
                    t_batch = torch.full((16,), t_, device=device, dtype=torch.long)
                    pred_noise = model(x, t_batch)
                    alpha_t = alphas[t_]
                    alpha_t_bar = alphas_cumprod[t_]
                    beta_t = betas[t_]
                    if t_ > 0:
                        noise = torch.randn_like(x)
                    else:
                        noise = 0
                    x = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_bar)) * pred_noise) + torch.sqrt(beta_t) * noise
                save_image(x, os.path.join(run_dir, f'sample_epoch_{epoch}.png'), normalize=True)
    print(f"Training complete. Check '{run_dir}' for generated samples and checkpoints.")

if __name__ == '__main__':
    main() 