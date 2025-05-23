import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from model import ConvVAE

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
    model = ConvVAE(latent_dim=latent_dim)
    
    # Load the best model checkpoint
    best_model_path = os.path.join('checkpoints', 'best_model', 'vae_best.pth')
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
        # Sample from standard normal distribution
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
        title = 'z=[\n' + '\n'.join(coord_groups) + ']'
        axes[i].set_title(title, fontsize=8, pad=10)
    
    plt.suptitle(f"Generated MNIST Samples (Latent Dim: {latent_dim})", fontsize=14)
    # Increase vertical space for the multi-line titles
    plt.tight_layout(rect=[0, 0, 1, 0.90], h_pad=2.0)  # Adjusted rect and added h_pad
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'generated_digits_{timestamp}.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Generated digits have been saved to '{output_path}'")

    # If latent space is 2D, also generate a grid of samples
    if latent_dim == 2:
        generate_latent_space_grid(model, device, output_dir, timestamp)

def generate_latent_space_grid(model, device, output_dir, timestamp, grid_size=20):
    """Generate a grid of samples by traversing the 2D latent space"""
    # Create a grid of latent vectors
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    z_grid = np.zeros((grid_size * grid_size, 2))
    
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            z_grid[i * grid_size + j] = [xi, yi]
    
    # Convert to tensor and generate samples
    with torch.no_grad():
        z = torch.FloatTensor(z_grid).to(device)
        samples = model.decode(z)
    
    # Create a large figure
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle("Latent Space Grid Visualization (2D)", fontsize=20)
    
    for i in range(grid_size * grid_size):
        ax = plt.subplot(grid_size, grid_size, i + 1)
        ax.imshow(samples[i][0].cpu(), cmap='gray')
        ax.axis('off')
        
        # Add coordinates as subtitle
        row = i // grid_size
        col = i % grid_size
        ax.set_title(f"({x[col]:.1f}, {y[row]:.1f})", fontsize=6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for suptitle
    output_path = os.path.join(output_dir, f'latent_space_grid_{timestamp}.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Latent space grid visualization has been saved to '{output_path}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate digits using trained VAE')
    parser.add_argument('--latent_dim', type=int, default=32, help='Dimension of latent space')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of digits to generate')

    args = parser.parse_args()
    generate_digits(
        latent_dim=args.latent_dim,
        num_samples=args.num_samples
    ) 