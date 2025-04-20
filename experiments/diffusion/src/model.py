import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class DiffusionModel(nn.Module):
    def __init__(self, in_channels=1, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial convolution
        self.conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)

        # Downsampling
        self.down1 = Block(64, 128, time_dim)
        self.sa1 = SelfAttention(128)
        self.down2 = Block(128, 256, time_dim)
        self.sa2 = SelfAttention(256)
        self.down3 = Block(256, 256, time_dim)
        self.sa3 = SelfAttention(256)

        # Bottleneck
        self.botconv1 = nn.Conv2d(256, 512, 3, padding=1)
        self.botconv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.botconv3 = nn.Conv2d(512, 256, 3, padding=1)

        # Upsampling
        self.up1 = Block(256, 128, time_dim, up=True)
        self.sa4 = SelfAttention(128)
        self.up2 = Block(128, 64, time_dim, up=True)
        self.sa5 = SelfAttention(64)
        self.up3 = Block(64, 64, time_dim, up=True)
        self.sa6 = SelfAttention(64)

        # Final convolution
        self.out = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, 1)
        )

    def forward(self, x, t):
        t = self.time_mlp(t)
        
        # Initial convolution
        x1 = self.conv0(x)
        
        # Downsampling path
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        
        # Bottleneck
        x4 = self.botconv1(x4)
        x4 = self.botconv2(x4)
        x4 = self.botconv3(x4)
        
        # Upsampling path with skip connections
        x = self.up1(x4, t)
        x = self.sa4(x)
        
        x = self.up2(x, t)
        x = self.sa5(x)
        
        x = self.up3(x, t)
        x = self.sa6(x)
        
        # Final convolution
        output = self.out(x)
        return output

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.mha(x, x, x)[0]
        x = x + self.ff_self(x)
        x = x.transpose(1, 2).view(-1, self.channels, *size)
        return x 