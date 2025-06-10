import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pixelcnn_prior import PixelCNN
import os

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices_path', type=str, default='code_indices.npy', help='Path to code indices .npy file')
    parser.add_argument('--num_embeddings', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_path', type=str, default='pixelcnn_prior.pth')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    # Load code indices
    indices = np.load(args.indices_path)
    indices = torch.LongTensor(indices)
    dataset = TensorDataset(indices)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # Model
    H, W = indices.shape[1:]
    model = PixelCNN(args.num_embeddings, input_shape=(H, W)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # Training
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        for (x,) in dataloader:
            x = x.to(device)  # (B, H, W)
            logits = model(x)  # (B, num_embeddings, H, W)
            loss = criterion(logits, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        if epoch % 5 == 0:
            torch.save({'model_state_dict': model.state_dict()}, args.save_path)
            print(f"Saved PixelCNN to {args.save_path}")
    torch.save({'model_state_dict': model.state_dict()}, args.save_path)
    print(f"Training complete. Final model saved to {args.save_path}")

if __name__ == '__main__':
    main() 