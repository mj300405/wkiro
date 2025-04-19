import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from model import DenoisingAE

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def generate_digits(latent_dim=32, num_samples=10):
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load the model
    model = DenoisingAE(latent_dim=latent_dim)
    
    # Load the best model checkpoint
    best_model_path = os.path.join('checkpoints', 'best_model', 'dae_best.pth')
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
        # Sample from normal distribution in the latent space
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decode(z)

    # Plot generated digits
    fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2.5))
    for i in range(num_samples):
        axes[i].imshow(samples[i][0].cpu(), cmap='gray')
        axes[i].axis('off')
        # Show latent space coordinates in a more readable format
        latent_coords = z[i].cpu().numpy()
        # Format coordinates with fewer decimals and group them
        coord_groups = []
        for j in range(0, len(latent_coords), 4):  # Show 4 coordinates per line
            group = latent_coords[j:j+4]
            coord_str = ', '.join([f'{coord:.1f}' for coord in group])
            coord_groups.append(coord_str)
        
        # Join groups with newlines and set as title
        title = 'h=[\n' + '\n'.join(coord_groups) + ']'
        axes[i].set_title(title, fontsize=8, pad=10)
    
    plt.suptitle(f"Generated MNIST Samples (Hidden Dim: {latent_dim})", fontsize=14)
    # Increase vertical space for the multi-line titles
    plt.tight_layout(rect=[0, 0, 1, 0.90], h_pad=2.0)
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'generated_digits_{timestamp}.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Generated digits have been saved to '{output_path}'")

def denoise_samples(latent_dim=32, num_samples=10, noise_factor=0.3):
    """Generate noisy samples and show denoised results."""
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load the model
    model = DenoisingAE(latent_dim=latent_dim)
    
    # Load the best model checkpoint
    best_model_path = os.path.join('checkpoints', 'best_model', 'dae_best.pth')
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

    # Generate random latent vectors and create noisy samples
    with torch.no_grad():
        # Create random images
        z = torch.randn(num_samples, latent_dim).to(device)
        clean_samples = model.decode(z)
        
        # Add noise and denoise
        noisy_samples = model.add_noise(clean_samples, noise_factor)
        denoised_samples, _ = model(clean_samples, noise_factor)

    # Plot samples
    fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
    for i in range(num_samples):
        # Clean samples
        axes[0, i].imshow(clean_samples[i][0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Clean')
            
        # Noisy samples
        axes[1, i].imshow(noisy_samples[i][0].cpu(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Noisy')
            
        # Denoised samples
        axes[2, i].imshow(denoised_samples[i][0].cpu(), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Denoised')
    
    plt.suptitle(f"Denoising Results (Noise Factor: {noise_factor})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'denoising_results_{timestamp}.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Denoising results have been saved to '{output_path}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate digits using trained Denoising AE')
    parser.add_argument('--latent_dim', type=int, default=32, help='Dimension of latent space')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of digits to generate')
    parser.add_argument('--noise_factor', type=float, default=0.3, help='Amount of noise to add (0-1)')
    parser.add_argument('--mode', choices=['generate', 'denoise'], default='generate',
                        help='Whether to generate new digits or show denoising results')

    args = parser.parse_args()
    if args.mode == 'generate':
        generate_digits(
            latent_dim=args.latent_dim,
            num_samples=args.num_samples
        )
    else:
        denoise_samples(
            latent_dim=args.latent_dim,
            num_samples=args.num_samples,
            noise_factor=args.noise_factor
        ) 