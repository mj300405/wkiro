import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
from model import Generator, Discriminator, weights_init

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

def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, epoch, g_loss, d_loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'g_loss': g_loss,
        'd_loss': d_loss
    }
    torch.save(checkpoint, filepath)

def save_samples(generator, epoch, device, latent_dim=100, n_samples=25, save_dir=None):
    """Generate and save sample images"""
    generator.eval()
    with torch.no_grad():
        # Generate random latent vectors
        z = torch.randn(n_samples, latent_dim).to(device)
        
        # Generate images
        fake_images = generator(z)
        
        # Create grid of images
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            img = fake_images[i].cpu().squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        
        # Save to generated_samples directory
        if save_dir is None:
            samples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'generated_samples')
        else:
            samples_dir = save_dir
            
        os.makedirs(samples_dir, exist_ok=True)
        plt.savefig(os.path.join(samples_dir, f'samples_epoch_{epoch}.png'))
        plt.close()

def train_gan(
    latent_dim=100,
    epochs=200,
    batch_size=128,
    lr=0.0002,
    beta1=0.5,
    save_interval=10,
    sample_interval=5,
    early_stopping_patience=10
):
    device = get_device()
    print(f"Using device: {device}")
    
    # Setup checkpoint directories
    best_model_dir, run_dir = setup_checkpoint_dirs()
    print(f"Saving checkpoints to: {run_dir}")
    print(f"Best model will be saved to: {best_model_dir}")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize models
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Training loop
    fixed_noise = torch.randn(64, latent_dim, device=device)  # For consistent samples
    real_label = 1
    fake_label = 0
    
    # For tracking best model
    best_g_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}/{epochs}')
        for batch_idx, (real_images, _) in enumerate(progress_bar):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            label_real = torch.full((batch_size,), real_label, device=device, dtype=torch.float32)
            label_fake = torch.full((batch_size,), fake_label, device=device, dtype=torch.float32)
            
            # Real images
            output_real = discriminator(real_images).view(-1)
            d_loss_real = criterion(output_real, label_real)
            
            # Fake images
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach()).view(-1)
            d_loss_fake = criterion(output_fake, label_fake)
            
            # Combined discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            output_fake = discriminator(fake_images).view(-1)
            g_loss = criterion(output_fake, label_real)  # We want to generate real
            g_loss.backward()
            g_optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}'
            })
            
            # Accumulate losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
        
        # Calculate average losses for the epoch
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        
        print(f'Epoch [{epoch}/{epochs}]')
        print(f'Generator Loss: {avg_g_loss:.4f}')
        print(f'Discriminator Loss: {avg_d_loss:.4f}')
        
        # Save samples
        #if epoch % sample_interval == 0:
        #    save_samples(generator, epoch, device, latent_dim, save_dir=run_dir)
        
        # Save checkpoint every save_interval epochs
        if epoch % save_interval == 0:
            checkpoint_path = os.path.join(run_dir, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, 
                           epoch, avg_g_loss, avg_d_loss, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model and early stopping
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            patience_counter = 0
            best_model_path = os.path.join(best_model_dir, 'gan_best.pth')
            save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, 
                           epoch, avg_g_loss, avg_d_loss, best_model_path)
            print(f"New best model saved with generator loss: {avg_g_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
    
    # Save final model
    final_checkpoint_path = os.path.join(run_dir, 'final.pt')
    save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, 
                   epochs, avg_g_loss, avg_d_loss, final_checkpoint_path)
    print(f"Final model saved: {final_checkpoint_path}")
    
    print("Training finished!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GAN on MNIST')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimension of latent space')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--save_interval', type=int, default=10, help='How often to save model checkpoints')
    parser.add_argument('--sample_interval', type=int, default=5, help='How often to save sample images')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    train_gan(
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        early_stopping_patience=args.early_stopping_patience
    ) 