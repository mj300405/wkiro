import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicAE(nn.Module):
    def __init__(self, hidden_dim=32):
        super(BasicAE, self).__init__()
        
        # Input dimensions for MNIST: 28x28 = 784
        self.input_dim = 784
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim * 8),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 8, self.input_dim),
            nn.Sigmoid()  # For MNIST pixel values between 0 and 1
        )
        
    def encode(self, x):
        # Flatten the input
        x = x.view(-1, self.input_dim)
        return self.encoder(x)
    
    def decode(self, z):
        x = self.decoder(z)
        # Reshape back to image dimensions
        return x.view(-1, 1, 28, 28)
    
    def forward(self, x):
        # Flatten input
        x_flat = x.view(-1, self.input_dim)
        # Encode
        z = self.encoder(x_flat)
        # Decode
        x_recon = self.decoder(z)
        # Reshape output
        return x_recon.view(-1, 1, 28, 28)
    
    def loss_function(self, recon_x, x):
        """
        Compute the reconstruction loss using binary cross entropy,
        which is better suited for image data than MSE.
        
        Args:
            recon_x: Reconstructed data
            x: Original data
            
        Returns:
            Reconstruction loss
        """
        x_flat = x.view(-1, self.input_dim)
        recon_x_flat = recon_x.view(-1, self.input_dim)
        BCE = F.binary_cross_entropy(recon_x_flat, x_flat, reduction='sum')
        
        return BCE 