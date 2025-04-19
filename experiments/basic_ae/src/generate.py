import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from model import BasicAE

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def generate_digits(hidden_dim=32, num_samples=10):
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load the model
    model = BasicAE(hidden_dim=hidden_dim)
    
    # Load the best model checkpoint
    best_model_path = os.path.join('checkpoints', 'best_model', 'ae_best.pth')
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

    # Generate random hidden vectors
    with torch.no_grad():
        # Sample from normal distribution in the hidden space
        z = torch.randn(num_samples, hidden_dim).to(device)
        samples = model.decode(z)

    # Plot generated digits
    fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2.5))
    for i in range(num_samples):
        axes[i].imshow(samples[i][0].cpu(), cmap='gray')
        axes[i].axis('off')
        # Show hidden space coordinates in a more readable format
        hidden_coords = z[i].cpu().numpy()
        # Format coordinates with fewer decimals and group them
        coord_groups = []
        for j in range(0, len(hidden_coords), 4):  # Show 4 coordinates per line
            group = hidden_coords[j:j+4]
            coord_str = ', '.join([f'{coord:.1f}' for coord in group])
            coord_groups.append(coord_str)
        
        # Join groups with newlines and set as title
        title = 'h=[\n' + '\n'.join(coord_groups) + ']'
        axes[i].set_title(title, fontsize=8, pad=10)
    
    plt.suptitle(f"Generated MNIST Samples (Hidden Dim: {hidden_dim})", fontsize=14)
    # Increase vertical space for the multi-line titles
    plt.tight_layout(rect=[0, 0, 1, 0.90], h_pad=2.0)
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'generated_digits_{timestamp}.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Generated digits have been saved to '{output_path}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate digits using trained Basic AE')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Dimension of hidden layer')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of digits to generate')

    args = parser.parse_args()
    generate_digits(
        hidden_dim=args.hidden_dim,
        num_samples=args.num_samples
    ) 