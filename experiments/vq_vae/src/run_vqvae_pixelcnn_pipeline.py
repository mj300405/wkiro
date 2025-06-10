import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import os
import argparse
from model import VQVAE
from pixelcnn_prior import PixelCNN

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def train_vqvae(num_embeddings, embedding_dim, commitment_cost, epochs, batch_size, learning_rate, early_stopping_patience, data_dir, checkpoint_dir):
    device = get_device()
    print(f"Using device: {device}")
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # Initialize model
    model = VQVAE(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, vq_loss, _ = model(data)
            loss, recon_loss, vq_loss = model.loss_function(recon_batch, data, vq_loss)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}")
        scheduler.step()
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon_batch, vq_loss, _ = model(data)
                loss, recon_loss, vq_loss = model.loss_function(recon_batch, data, vq_loss)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Val Loss: {avg_val_loss:.4f}")
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'vqvae_epoch_{epoch+1}.pth')
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_path = os.path.join(checkpoint_dir, 'vqvae_best.pth')
            torch.save({'model_state_dict': model.state_dict()}, best_model_path)
            print(f"New best model saved with validation loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    return model

def extract_code_indices(model, dataloader, device):
    all_indices = []
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            _, _, indices = model(x)
            all_indices.append(indices.cpu().numpy())
    all_indices = np.concatenate(all_indices, axis=0)
    return all_indices

def train_pixelcnn(indices, num_embeddings, epochs, batch_size, learning_rate, save_path, device):
    indices = torch.LongTensor(indices)
    dataset = TensorDataset(indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    H, W = indices.shape[1:]
    model = PixelCNN(num_embeddings, input_shape=(H, W)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for (x,) in dataloader:
            x = x.to(device)
            logits = model(x)
            loss = criterion(logits, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")
        if epoch % 5 == 0:
            torch.save({'model_state_dict': model.state_dict()}, save_path)
            print(f"Saved PixelCNN to {save_path}")
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"Training complete. Final model saved to {save_path}")
    return model

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

def generate_samples(pixelcnn, vqvae, num_samples, shape, device, output_dir):
    samples = sample_pixelcnn(pixelcnn, num_samples, shape, device)
    vqvae.eval()
    with torch.no_grad():
        x_recon = vqvae.decode_from_indices(samples)
    os.makedirs(output_dir, exist_ok=True)
    # Save all samples in a single grid image
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    grid = torch.zeros((1, grid_size * 28, grid_size * 28), device=device)
    for i, img in enumerate(x_recon):
        row = i // grid_size
        col = i % grid_size
        grid[0, row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] = img[0]
    save_image(grid, os.path.join(output_dir, 'all_samples.png'))
    print(f"Saved {num_samples} samples in a single grid image to {output_dir}/all_samples.png")

def main(args):
    device = get_device()
    print(f"Using device: {device}")
    # Train VQ-VAE
    print("Training VQ-VAE...")
    vqvae = train_vqvae(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        commitment_cost=args.commitment_cost,
        epochs=args.vqvae_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    # Extract code indices
    print("Extracting code indices...")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    indices = extract_code_indices(vqvae, dataloader, device)
    indices_path = os.path.join(args.checkpoint_dir, 'code_indices.npy')
    np.save(indices_path, indices)
    print(f"Saved code indices to {indices_path}, shape: {indices.shape}")
    # Train PixelCNN
    print("Training PixelCNN...")
    pixelcnn = train_pixelcnn(
        indices=indices,
        num_embeddings=args.num_embeddings,
        epochs=args.pixelcnn_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_path=os.path.join(args.checkpoint_dir, 'pixelcnn_prior.pth'),
        device=device
    )
    # Generate samples
    print("Generating samples...")
    generate_samples(
        pixelcnn=pixelcnn,
        vqvae=vqvae,
        num_samples=args.num_samples,
        shape=(7, 7),  # Default for MNIST 28x28 with 4x downsampling
        device=device,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run VQ-VAE + PixelCNN pipeline')
    parser.add_argument('--num_embeddings', type=int, default=512, help='Number of embeddings in codebook')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of each embedding')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='Commitment cost for VQ loss')
    parser.add_argument('--vqvae_epochs', type=int, default=100, help='Number of epochs to train VQ-VAE')
    parser.add_argument('--pixelcnn_epochs', type=int, default=20, help='Number of epochs to train PixelCNN')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to save MNIST data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--output_dir', type=str, default='generated_samples', help='Directory to save generated samples')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    args = parser.parse_args()
    main(args) 