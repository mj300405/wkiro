import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(ConvVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (batch_size, 1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (batch_size, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (batch_size, 64, 7, 7)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # -> (batch_size, 128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Flatten(),  # -> (batch_size, 128 * 7 * 7)
        )
        
        # Calculate the flattened size
        self.flattened_size = 128 * 7 * 7  # 6272
        
        # Latent space
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)
        
        self.decoder = nn.Sequential(
            # Start with: (batch_size, 128, 7, 7)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (batch_size, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> (batch_size, 32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),  # -> (batch_size, 1, 28, 28)
            nn.Sigmoid()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 128, 7, 7)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var
    
    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        """
        Compute the VAE loss function.
        
        Args:
            recon_x: Reconstructed data
            x: Original data
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
            beta: Weight for the KL divergence term (for beta-VAE)
            
        Returns:
            Total loss, reconstruction loss, and KL divergence
        """
        # Reconstruction loss (binary cross entropy)
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        loss = BCE + beta * KLD
        
        return loss, BCE, KLD 