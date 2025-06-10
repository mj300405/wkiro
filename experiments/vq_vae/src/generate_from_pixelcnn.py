import torch
import numpy as np
from torchvision.utils import save_image
from pixelcnn_prior import PixelCNN
from model import VQVAE
import os

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def sample_pixelcnn(model, num_samples, shape, device):
    model.eval()
    samples = torch.zeros((num_samples, *shape), dtype=torch.long, device=device)
    with torch.no_grad():
        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = model(samples)
                probs = torch.softmax(logits[:,:,i,j], dim=1)
                samples[:,i,j] = torch.multinomial(probs, 1).squeeze(-1)
    return samples

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pixelcnn_path', type=str, required=True)
    parser.add_argument('--vqvae_path', type=str, required=True)
    parser.add_argument('--num_embeddings', type=int, default=512)
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='pixelcnn_samples')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    # Load PixelCNN
    dummy_shape = (7, 7)  # Default for MNIST 28x28 with 4x downsampling
    pixelcnn = PixelCNN(args.num_embeddings, input_shape=dummy_shape).to(device)
    pixelcnn_ckpt = torch.load(args.pixelcnn_path, map_location=device)
    pixelcnn.load_state_dict(pixelcnn_ckpt['model_state_dict'])
    # Load VQ-VAE
    vqvae = VQVAE().to(device)
    vqvae_ckpt = torch.load(args.vqvae_path, map_location=device)
    vqvae.load_state_dict(vqvae_ckpt['model_state_dict'])
    # Sample code indices
    samples = sample_pixelcnn(pixelcnn, args.num_samples, dummy_shape, device)  # (N, H, W)
    # Decode
    vqvae.eval()
    with torch.no_grad():
        x_recon = vqvae.decode_from_indices(samples)
    os.makedirs(args.output_dir, exist_ok=True)
    for i, img in enumerate(x_recon):
        save_image(img, os.path.join(args.output_dir, f'sample_{i}.png'))
    print(f"Saved {args.num_samples} samples to {args.output_dir}")

if __name__ == '__main__':
    main() 