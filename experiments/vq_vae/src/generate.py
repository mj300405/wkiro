import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from model import VQVAE
import os
import argparse

def load_model(checkpoint_path, device, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
    model = VQVAE(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def visualize_codebook(model, save_path):
    """Visualize the learned codebook embeddings."""
    embeddings = model.vq.embedding.weight.data.cpu()
    
    # Reshape embeddings to square grid
    n = int(np.ceil(np.sqrt(len(embeddings))))
    grid_size = n * n
    
    # Pad with zeros if necessary
    if grid_size > len(embeddings):
        padding = grid_size - len(embeddings)
        embeddings = torch.cat([embeddings, torch.zeros(padding, embeddings.size(1))])
    
    # Reshape to square grid
    grid = embeddings.view(n, n, -1)
    
    # Plot using imshow
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.norm(dim=2), cmap='viridis')
    plt.colorbar(label='L2 Norm of Embedding Vectors')
    plt.title('Codebook Embeddings Visualization')
    plt.savefig(save_path)
    plt.close()

def interpolate_latent(model, img1, img2, num_steps=10, save_path=None):
    """Interpolate between two images in the latent space."""
    device = next(model.parameters()).device  # Get the device the model is on
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    with torch.no_grad():
        # Get latent representations
        _, _, indices1 = model(img1.unsqueeze(0))
        _, _, indices2 = model(img2.unsqueeze(0))
        
        # Convert indices to one-hot vectors
        one_hot1 = F.one_hot(indices1.view(-1), num_classes=model.vq.num_embeddings).float().to(device)
        one_hot2 = F.one_hot(indices2.view(-1), num_classes=model.vq.num_embeddings).float().to(device)
        
        # Create interpolation steps
        alphas = torch.linspace(0, 1, num_steps, device=device)
        interpolated_images = []
        
        for alpha in alphas:
            # Interpolate one-hot vectors
            interp = (1 - alpha) * one_hot1 + alpha * one_hot2
            
            # Get embeddings
            quantized = torch.matmul(interp, model.vq.embedding.weight)
            quantized = quantized.view(indices1.shape[0], -1, 7, 7)  # Reshape to [batch_size, channels, height, width]
            
            # Decode
            decoded = model.decoder(quantized)
            interpolated_images.append(decoded.squeeze(0))
    
    # Visualize interpolation
    fig, axes = plt.subplots(1, num_steps, figsize=(20, 2))
    for i, img in enumerate(interpolated_images):
        axes[i].imshow(img[0].cpu(), cmap='gray')
        axes[i].axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
    
    return interpolated_images

def generate_samples(model, num_samples=10, temperature=1.0, save_path=None):
    """Generate new samples by sampling from the codebook."""
    model.eval()
    device = next(model.parameters()).device  # Get the device the model is on
    with torch.no_grad():
        # Sample random indices
        probs = torch.ones(model.vq.num_embeddings, device=device) / model.vq.num_embeddings
        indices = torch.multinomial(probs, num_samples * 7 * 7, replacement=True)  # 7x7 is the encoded image size
        indices = indices.view(num_samples, 1, 7, 7)  # Reshape to match the expected format
        
        # Get embeddings for these indices
        one_hot = F.one_hot(indices.view(-1), num_classes=model.vq.num_embeddings).float().to(device)
        quantized = torch.matmul(one_hot, model.vq.embedding.weight)
        # Reshape to [batch_size, channels, height, width]
        quantized = quantized.view(num_samples, -1, 7, 7)
        
        # Decode
        samples = model.decoder(quantized)
    
    # Visualize samples
    fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
    for i, sample in enumerate(samples):
        axes[i].imshow(sample[0].cpu(), cmap='gray')
        axes[i].axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
    
    return samples

def main(args):
    device = get_device()
    print(f"Using device: {device}")
    
    # Load the trained model
    model = load_model(
        args.checkpoint_path,
        device,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        commitment_cost=args.commitment_cost
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize codebook
    codebook_path = os.path.join(args.output_dir, 'codebook.png')
    visualize_codebook(model, codebook_path)
    print(f"Codebook visualization saved to {codebook_path}")
    
    # Generate samples
    samples_path = os.path.join(args.output_dir, 'generated_samples.png')
    generate_samples(model, num_samples=args.num_samples, temperature=args.temperature, save_path=samples_path)
    print(f"Generated samples saved to {samples_path}")
    
    # Load some test images for interpolation
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    img1, _ = test_dataset[0]
    img2, _ = test_dataset[1]
    
    # Interpolate between test images
    interpolation_path = os.path.join(args.output_dir, 'interpolation.png')
    interpolate_latent(model, img1, img2, num_steps=args.interpolation_steps, save_path=interpolation_path)
    print(f"Interpolation results saved to {interpolation_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate samples from trained VQ-VAE')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_model/vqvae_best.pth', help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='generated_samples', help='Directory to save generated images')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--interpolation_steps', type=int, default=10, help='Number of interpolation steps')
    parser.add_argument('--num_embeddings', type=int, default=512, help='Number of embeddings in codebook')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of each embedding')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='Commitment cost for VQ loss')
    
    args = parser.parse_args()
    main(args) 