import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(DenoisingAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (batch_size, 1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (batch_size, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (batch_size, 64, 7, 7)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # -> (batch_size, 128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),  # -> (batch_size, 128 * 7 * 7)
        )
        
        # Calculate the flattened size
        self.flattened_size = 128 * 7 * 7  # 6272
        
        # Latent space
        self.fc = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            # Start with: (batch_size, 128, 7, 7)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (batch_size, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> (batch_size, 32, 28, 28)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),  # -> (batch_size, 1, 28, 28)
            nn.Sigmoid()
        )
        
    def add_noise(self, x, noise_factor=0.3):
        """Add random noise to the input."""
        noise = torch.randn_like(x) * noise_factor
        noisy_x = x + noise
        return torch.clamp(noisy_x, 0., 1.)
    
    def encode(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 128, 7, 7)
        x = self.decoder(x)
        return x
    
    def forward(self, x, noise_factor=0.3):
        # Add noise to input
        noisy_x = self.add_noise(x, noise_factor)
        # Encode
        x = self.encoder(noisy_x)
        # Latent space
        z = self.fc(x)
        # Decode
        x_recon = self.decode(z)
        return x_recon, noisy_x
    
    def loss_function(self, recon_x, x):
        """
        Compute the reconstruction loss using binary cross entropy.
        Note: We compute loss against the original clean image, not the noisy one.
        
        Args:
            recon_x: Reconstructed data
            x: Original clean data
            
        Returns:
            Reconstruction loss
        """
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        return BCE 