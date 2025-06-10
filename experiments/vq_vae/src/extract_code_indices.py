import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from model import VQVAE

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def extract_indices(model, dataloader, device):
    all_indices = []
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            _, _, indices = model(x)
            # indices: (B, H, W)
            all_indices.append(indices.cpu().numpy())
    all_indices = np.concatenate(all_indices, axis=0)
    return all_indices

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to trained VQ-VAE checkpoint')
    parser.add_argument('--output_path', type=str, default='code_indices.npy', help='Where to save the code indices')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    # Load model
    model = VQVAE()
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    # Data
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST('../../data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    # Extract indices
    indices = extract_indices(model, dataloader, device)
    np.save(args.output_path, indices)
    print(f"Saved code indices to {args.output_path}, shape: {indices.shape}")

if __name__ == '__main__':
    main() 