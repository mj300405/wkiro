import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from model import ConditionalVAE

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def generate_specific_digits(latent_dim=32, num_samples_per_digit=3):
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load the model
    model = ConditionalVAE(latent_dim=latent_dim)
    
    # Load the best model checkpoint
    best_model_path = os.path.join('checkpoints', 'best_model', 'cvae_best.pth')
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

    # Generate samples for each digit
    with torch.no_grad():
        # Create a figure with 10 rows (one for each digit) and num_samples_per_digit columns
        fig, axes = plt.subplots(10, num_samples_per_digit, figsize=(2*num_samples_per_digit, 20))
        
        for digit in range(10):
            # Create labels tensor
            labels = torch.tensor([digit] * num_samples_per_digit).to(device)
            
            # Sample random latent vectors
            z = torch.randn(num_samples_per_digit, latent_dim).to(device)
            
            # Generate samples
            samples = model.decode(z, labels)
            
            # Plot the samples
            for j in range(num_samples_per_digit):
                axes[digit, j].imshow(samples[j][0].cpu(), cmap='gray')
                axes[digit, j].axis('off')
                if j == 0:
                    axes[digit, j].set_title(f'Digit {digit}', fontsize=12)
    
    plt.suptitle(f"Generated MNIST Digits (Latent Dim: {latent_dim})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for suptitle
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'generated_digits_{timestamp}.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Generated digits have been saved to '{output_path}'")

def generate_digit_interpolation(start_digit, end_digit, num_steps=10, latent_dim=32):
    """Generate interpolation between two digits"""
    # Set device
    device = get_device()
    
    # Load the model
    model = ConditionalVAE(latent_dim=latent_dim)
    
    # Load the best model checkpoint
    best_model_path = os.path.join('checkpoints', 'best_model', 'cvae_best.pth')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"No best model found at {best_model_path}. Please train the model first.")
    
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create output directory if it doesn't exist
    output_dir = 'generated_samples'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        # Create a fixed latent vector
        z = torch.randn(1, latent_dim).to(device)
        
        # Create interpolated labels
        alphas = np.linspace(0, 1, num_steps)
        
        # Create figure
        fig, axes = plt.subplots(1, num_steps, figsize=(2*num_steps, 2.5))
        
        for i, alpha in enumerate(alphas):
            # Interpolate between the two digits
            label = alpha * end_digit + (1 - alpha) * start_digit
            label_tensor = torch.tensor([label]).to(device)
            
            # Generate sample
            sample = model.decode(z, label_tensor)
            
            # Plot
            axes[i].imshow(sample[0][0].cpu(), cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'{label:.1f}', fontsize=10)

        plt.suptitle(f'Interpolation from {start_digit} to {end_digit}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'interpolation_{start_digit}_to_{end_digit}_{timestamp}.png')
        plt.savefig(output_path)
        plt.close()

        print(f"Interpolation has been saved to '{output_path}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate digits using trained Conditional VAE')
    parser.add_argument('--latent_dim', type=int, default=32, help='Dimension of latent space')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples per digit')
    parser.add_argument('--interpolate', action='store_true', help='Generate digit interpolation')
    parser.add_argument('--start_digit', type=int, default=0, help='Start digit for interpolation')
    parser.add_argument('--end_digit', type=int, default=9, help='End digit for interpolation')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of interpolation steps')

    args = parser.parse_args()
    
    if args.interpolate:
        generate_digit_interpolation(
            start_digit=args.start_digit,
            end_digit=args.end_digit,
            num_steps=args.num_steps,
            latent_dim=args.latent_dim
        )
    else:
        generate_specific_digits(
            latent_dim=args.latent_dim,
            num_samples_per_digit=args.num_samples
        ) 