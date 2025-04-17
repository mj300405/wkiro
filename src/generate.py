import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from model import Autoencoder, VariationalAutoencoder

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def generate_digits(model_type='standard', latent_dim=32, num_samples=10):
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load the model
    if model_type == 'standard':
        model = Autoencoder(latent_dim=latent_dim)
    else:
        model = VariationalAutoencoder(latent_dim=latent_dim)
    
    # Load the best model checkpoint
    best_model_path = os.path.join('checkpoints', 'best_model', f'autoencoder_{model_type}_best.pth')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"No best model found at {best_model_path}. Please train the model first.")
    
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with loss: {checkpoint['loss']:.4f}")
    
    model = model.to(device)
    model.eval()

    # Create output directory if it doesn't exist
    output_dir = 'generated_samples'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate random latent vectors
    with torch.no_grad():
        # For standard autoencoder, sample from normal distribution
        # For VAE, sample from standard normal distribution
        z = torch.randn(num_samples, latent_dim).to(device)
        
        if model_type == 'standard':
            samples = model.decoder(z)
        else:
            samples = model.decode(z)
        
        samples = samples.view(-1, 1, 28, 28)

    # Plot generated digits
    fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
    for i in range(num_samples):
        axes[i].imshow(samples[i][0].cpu(), cmap='gray')
        axes[i].axis('off')
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'generated_digits_{model_type}_{timestamp}.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Generated digits have been saved to '{output_path}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate digits using trained autoencoder')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'variational'],
                        help='Type of autoencoder to use (standard or variational)')
    parser.add_argument('--latent_dim', type=int, default=32, help='Dimension of latent space')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of digits to generate')

    args = parser.parse_args()
    generate_digits(
        model_type=args.model_type,
        latent_dim=args.latent_dim,
        num_samples=args.num_samples
    ) 