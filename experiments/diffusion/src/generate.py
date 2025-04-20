import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os
from model import DiffusionModel
from train import DiffusionTrainer

# Set device
device = (
    "mps" 
    if torch.backends.mps.is_available()
    else "cuda" 
    if torch.cuda.is_available() 
    else "cpu"
)
print(f"Using device: {device}")

def sample(model, trainer, n_samples=16, device=device):
    model.eval()
    with torch.no_grad():
        # Start from pure noise (single channel for MNIST)
        x = torch.randn(n_samples, 1, 32, 32).to(device)
        
        # Iteratively denoise
        for t in reversed(range(trainer.timesteps)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_batch)
            alpha_t = trainer.alphas[t]
            alpha_t_bar = trainer.alphas_cumprod[t]
            beta_t = trainer.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_bar)) * predicted_noise) + \
                torch.sqrt(beta_t) * noise
    
    return x

def main():
    # Initialize model and trainer
    model = DiffusionModel(in_channels=1)  # Single channel for MNIST
    trainer = DiffusionTrainer(model)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/checkpoint_epoch_100.pt'  # Adjust path as needed
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded checkpoint successfully")
    else:
        print("No checkpoint found. Using untrained model.")
    
    # Generate samples
    samples = sample(model, trainer)
    
    # Save samples
    os.makedirs('generated_samples', exist_ok=True)
    save_image(samples, 'generated_samples/samples.png', normalize=True)
    print("Samples saved to generated_samples/samples.png")

if __name__ == '__main__':
    main() 