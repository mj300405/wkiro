import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=32, num_classes=10):
        super(ConditionalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
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
        
        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, 32)
        
        # Combine flattened input with label embedding
        self.combine_inputs = nn.Sequential(
            nn.Linear(self.flattened_size + 32, self.flattened_size),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim + 32, self.flattened_size),
            nn.ReLU()
        )
        
        # Decoder
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
        
    def encode(self, x, labels):
        # Encode image
        x = self.encoder(x)
        
        # Get label embedding
        label_embedding = self.label_embedding(labels)
        
        # Combine image features with label embedding
        combined = torch.cat([x, label_embedding], dim=1)
        combined = self.combine_inputs(combined)
        
        # Get latent parameters
        mu = self.fc_mu(combined)
        log_var = self.fc_var(combined)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels):
        # Get label embedding
        label_embedding = self.label_embedding(labels)
        
        # Combine latent vector with label embedding
        z = torch.cat([z, label_embedding], dim=1)
        
        # Decode
        x = self.decoder_input(z)
        x = x.view(-1, 128, 7, 7)
        x = self.decoder(x)
        return x
    
    def forward(self, x, labels):
        mu, log_var = self.encode(x, labels)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z, labels)
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