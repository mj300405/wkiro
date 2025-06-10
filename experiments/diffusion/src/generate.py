import torch
import torch.nn as nn
import os
from torchvision.utils import save_image
import imageio
from model import DiffusionModel
import glob

# Set device
device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

def find_latest_checkpoint():
    run_dirs = sorted(glob.glob('checkpoints/run_*'), reverse=True)
    if not run_dirs:
        return None
    ckpts = sorted(glob.glob(os.path.join(run_dirs[0], 'checkpoint_epoch_*.pt')), reverse=True)
    if not ckpts:
        return None
    return ckpts[0]

def sample(model, checkpoint_path, n_samples=16, timesteps=1000, save_gif_path=None):
    # DDPM schedule
    beta_start = 1e-4
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.eval()
    frames = []
    with torch.no_grad():
        x = torch.randn(n_samples, 1, 32, 32).to(device)
        for t_ in reversed(range(timesteps)):
            t_batch = torch.full((n_samples,), t_, device=device, dtype=torch.long)
            pred_noise = model(x, t_batch)
            alpha_t = alphas[t_]
            alpha_t_bar = alphas_cumprod[t_]
            beta_t = betas[t_]
            if t_ > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            x = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_bar)) * pred_noise) + torch.sqrt(beta_t) * noise
            # Save first image for GIF
            frame = x[0].detach().cpu().numpy()
            frame = (frame + 1) / 2
            frame = (frame * 255).clip(0, 255).astype('uint8').squeeze(0)
            frames.append(frame)
    if save_gif_path is not None:
        imageio.mimsave(save_gif_path, frames, duration=0.05)
    return x

def main():
    model = DiffusionModel(in_channels=1, base_ch=32).to(device)
    checkpoint_path = find_latest_checkpoint()
    if checkpoint_path is None:
        print("No checkpoint found in checkpoints/run_*/checkpoint_epoch_*.pt")
        return
    os.makedirs('generated_samples', exist_ok=True)
    samples = sample(model, checkpoint_path, save_gif_path='generated_samples/denoising.gif')
    save_image(samples, 'generated_samples/samples.png', normalize=True)
    print("Samples saved to generated_samples/samples.png")
    print("Denoising GIF saved to generated_samples/denoising.gif")

if __name__ == '__main__':
    main() 