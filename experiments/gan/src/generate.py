import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from datetime import datetime
from model import Generator
from train import get_device

def load_generator(checkpoint_path, latent_dim=100):
    """Load a trained generator from checkpoint"""
    device = get_device()
    generator = Generator(latent_dim).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    return generator, device

def generate_grid(generator, device, latent_dim=100, n_rows=5, n_cols=5):
    """Generate a grid of samples from the generator"""
    with torch.no_grad():
        # Generate random latent vectors
        n_samples = n_rows * n_cols
        z = torch.randn(n_samples, latent_dim).to(device)
        
        # Generate images
        fake_images = generator(z)
        
        # Create grid of images
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            img = fake_images[i].cpu().squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        
        return fig

def generate_samples(latent_dim=100, n_samples=25):
    """Generate samples from the best trained model"""
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load the generator
    generator = Generator(latent_dim).to(device)
    
    # Load the best model checkpoint
    best_model_path = os.path.join('checkpoints', 'best_model', 'gan_best.pth')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"No best model found at {best_model_path}. Please train the model first.")
    
    checkpoint = torch.load(best_model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with generator loss: {checkpoint['g_loss']:.4f}")
    
    generator.eval()
    
    # Create output directory if it doesn't exist
    output_dir = 'generated_samples'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate and save samples
    fig = generate_grid(generator, device, latent_dim, n_rows=5, n_cols=5)
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'generated_grid_{timestamp}.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Generated samples have been saved to '{output_path}'")

def generate_interpolation(start_z, end_z, num_steps=10, latent_dim=100):
    """Generate interpolation between two latent vectors"""
    # Set device
    device = get_device()
    
    # Load the generator
    generator = Generator(latent_dim).to(device)
    
    # Load the best model checkpoint
    best_model_path = os.path.join('checkpoints', 'best_model', 'gan_best.pth')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"No best model found at {best_model_path}. Please train the model first.")
    
    checkpoint = torch.load(best_model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Create output directory if it doesn't exist
    output_dir = 'generated_samples'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with torch.no_grad():
        # Create interpolated latent vectors
        alphas = np.linspace(0, 1, num_steps)
        z_interp = torch.zeros(num_steps, latent_dim).to(device)
        
        for i, alpha in enumerate(alphas):
            z_interp[i] = alpha * end_z + (1 - alpha) * start_z
        
        # Generate samples
        samples = generator(z_interp)
        
        # Create figure
        fig, axes = plt.subplots(1, num_steps, figsize=(2*num_steps, 2.5))
        
        for i in range(num_steps):
            axes[i].imshow(samples[i][0].cpu(), cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Step {i+1}', fontsize=10)
        
        plt.suptitle('Latent Space Interpolation', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'interpolation_{timestamp}.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Interpolation has been saved to '{output_path}'")

def main():
    parser = argparse.ArgumentParser(description='Generate samples using trained GAN')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimension of latent space')
    parser.add_argument('--n_samples', type=int, default=25, help='Number of samples to generate')
    parser.add_argument('--interpolate', action='store_true', help='Generate latent space interpolation')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of interpolation steps')
    
    args = parser.parse_args()
    
    if args.interpolate:
        # Generate random start and end points in latent space
        start_z = torch.randn(1, args.latent_dim)
        end_z = torch.randn(1, args.latent_dim)
        generate_interpolation(
            start_z=start_z,
            end_z=end_z,
            num_steps=args.num_steps,
            latent_dim=args.latent_dim
        )
    else:
        generate_samples(
            latent_dim=args.latent_dim,
            n_samples=args.n_samples
        )

if __name__ == "__main__":
    main() 