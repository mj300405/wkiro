import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
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

def setup_checkpoint_dirs():
    # Create main checkpoints directory
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    # Create best model directory
    best_model_dir = os.path.join('checkpoints', 'best_model')
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('checkpoints', f'run_{timestamp}')
    os.makedirs(run_dir)
    
    return best_model_dir, run_dir

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

def train_vae(latent_dim=2, epochs=500, batch_size=128, learning_rate=1e-4):
    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # Setup checkpoint directories
    best_model_dir, run_dir = setup_checkpoint_dirs()
    print(f"Saving checkpoints to: {run_dir}")
    print(f"Best model will be saved to: {best_model_dir}")

    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = ConvVAE(latent_dim=latent_dim)
    model = model.to(device)

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    # Training loop
    train_losses = []
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        # Calculate KL weight with warm-up
        kl_weight = min(1.0, (epoch + 1) / 50) * 0.01  # Gradual increase over 50 epochs

        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = model(data)
            # Reconstruction loss
            recon_loss = criterion(recon_batch, data)
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            # Total loss with warm-up KL weight
            loss = recon_loss + kl_weight * kl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

        # Step the scheduler
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(run_dir, f'epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model if current loss is better
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(best_model_dir, 'vae_best.pth')
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, best_model_path)
            print(f"New best model saved with loss: {avg_loss:.4f}")

        # Save sample reconstructions
        if (epoch + 1) % 10 == 0:
            save_reconstructions(model, data[:8], epoch + 1, run_dir)

    # Save final model
    final_checkpoint_path = os.path.join(run_dir, 'final.pth')
    save_checkpoint(model, optimizer, epochs, avg_loss, final_checkpoint_path)
    print(f"Final model saved: {final_checkpoint_path}")

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(run_dir, 'training_loss.png'))
    plt.close()

def save_reconstructions(model, data, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        recon_batch, _, _ = model(data)

    # Plot original and reconstructed images
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        # Original images
        axes[0, i].imshow(data[i][0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')

        # Reconstructed images
        axes[1, i].imshow(recon_batch[i][0].cpu(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')

    plt.savefig(os.path.join(save_dir, f'reconstruction_epoch_{epoch}.png'))
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Convolutional VAE on MNIST')
    parser.add_argument('--latent_dim', type=int, default=2, help='Dimension of latent space')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')

    args = parser.parse_args()
    train_vae(
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    ) 