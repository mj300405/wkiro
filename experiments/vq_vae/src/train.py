import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from datetime import datetime
from model import VQVAE

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def setup_checkpoint_dirs():
    # Create checkpoint directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('checkpoints', f'run_{timestamp}')
    best_model_dir = os.path.join('checkpoints', 'best_model')
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    
    return best_model_dir, run_dir

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def train_vqvae(num_embeddings=512, embedding_dim=64, commitment_cost=0.25, epochs=100, 
                batch_size=128, learning_rate=1e-3, early_stopping_patience=10):
    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # Setup checkpoint directories
    best_model_dir, run_dir = setup_checkpoint_dirs()
    print(f"Saving checkpoints to: {run_dir}")
    print(f"Best model will be saved to: {best_model_dir}")

    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load and split dataset
    full_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = VQVAE(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
    model = model.to(device)

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Training loop
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_recon_loss = 0
        total_train_vq_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, vq_loss, _ = model(data)
            loss, recon_loss, vq_loss = model.loss_function(recon_batch, data, vq_loss)
            
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_recon_loss += recon_loss.item()
            total_train_vq_loss += vq_loss.item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'recon_loss': recon_loss.item(),
                'vq_loss': vq_loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })

        # Validation phase
        model.eval()
        total_val_loss = 0
        total_val_recon_loss = 0
        total_val_vq_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon_batch, vq_loss, _ = model(data)
                loss, recon_loss, vq_loss = model.loss_function(recon_batch, data, vq_loss)
                
                total_val_loss += loss.item()
                total_val_recon_loss += recon_loss.item()
                total_val_vq_loss += vq_loss.item()

        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_train_recon_loss = total_train_recon_loss / len(train_loader.dataset)
        avg_train_vq_loss = total_train_vq_loss / len(train_loader.dataset)
        
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        avg_val_recon_loss = total_val_recon_loss / len(val_loader.dataset)
        avg_val_vq_loss = total_val_vq_loss / len(val_loader.dataset)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon_loss:.4f}, VQ: {avg_train_vq_loss:.4f})')
        print(f'Val Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon_loss:.4f}, VQ: {avg_val_vq_loss:.4f})')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # Step the scheduler
        scheduler.step()

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(run_dir, f'epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, avg_val_loss, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model and early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_path = os.path.join(best_model_dir, 'vqvae_best.pth')
            save_checkpoint(model, optimizer, epoch + 1, avg_val_loss, best_model_path)
            print(f"New best model saved with validation loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Save sample reconstructions
        if (epoch + 1) % 10 == 0:
            save_reconstructions(model, data[:8], epoch + 1, run_dir)

    # Save final model
    final_checkpoint_path = os.path.join(run_dir, 'final.pth')
    save_checkpoint(model, optimizer, epochs, avg_val_loss, final_checkpoint_path)
    print(f"Final model saved: {final_checkpoint_path}")

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'training_loss.png'))
    plt.close()

def save_reconstructions(model, data, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        recon_batch, _, indices = model(data)

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

        # Add codebook indices as text
        unique_indices = torch.unique(indices[i])
        axes[1, i].set_title(f'Used {len(unique_indices)} codes', fontsize=8)

    plt.savefig(os.path.join(save_dir, f'reconstruction_epoch_{epoch}.png'))
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train VQ-VAE on MNIST')
    parser.add_argument('--num_embeddings', type=int, default=512, help='Number of embeddings in codebook')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of each embedding')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='Commitment cost for VQ loss')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')

    args = parser.parse_args()
    train_vqvae(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        commitment_cost=args.commitment_cost,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience
    ) 