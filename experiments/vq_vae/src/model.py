import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Create the embedding table
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        
    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()  # Straight-through estimator
        
        # Convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, encoding_indices.view(input_shape[:-1])

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (batch_size, 1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (batch_size, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (batch_size, 64, 7, 7)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=1, padding=1),  # -> (batch_size, embedding_dim, 7, 7)
            nn.BatchNorm2d(embedding_dim),
            nn.LeakyReLU(0.2),
        )
        
        # Vector Quantization layer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # Decoder
        self.decoder = nn.Sequential(
            # Input: (batch_size, embedding_dim, 7, 7)
            nn.ConvTranspose2d(embedding_dim, 64, kernel_size=4, stride=2, padding=1),  # -> (batch_size, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> (batch_size, 32, 28, 28)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),  # -> (batch_size, 1, 28, 28)
            nn.Sigmoid()
        )
        
    def encode(self, x):
        z = self.encoder(x)
        z_q, _, indices = self.vq(z)
        return z_q, indices
    
    def decode(self, z_q):
        return self.decoder(z_q)
    
    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices
    
    def loss_function(self, recon_x, x, vq_loss):
        """
        Compute the VQ-VAE loss, which consists of:
        1. Reconstruction loss (binary cross entropy)
        2. Vector quantization loss (from the VQ layer)
        
        Args:
            recon_x: Reconstructed data
            x: Original data
            vq_loss: Loss from vector quantization
            
        Returns:
            Total loss, reconstruction loss, and VQ loss
        """
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        total_loss = recon_loss + vq_loss
        
        return total_loss, recon_loss, vq_loss 