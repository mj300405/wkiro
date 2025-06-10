import torch
import torch.nn as nn

# Minimal U-Net block
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.conv(x)

class DiffusionModel(nn.Module):
    def __init__(self, in_channels=1, base_ch=32):
        super().__init__()
        self.enc1 = UNetBlock(in_channels, base_ch)
        self.enc2 = UNetBlock(base_ch, base_ch*2)
        self.enc3 = UNetBlock(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool2d(2)
        self.middle = UNetBlock(base_ch*4, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = UNetBlock(base_ch*2 + base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = UNetBlock(base_ch + base_ch*2, base_ch)
        self.up0 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec0 = UNetBlock(base_ch + base_ch, base_ch)
        self.outc = nn.Conv2d(base_ch, in_channels, 1)
    def forward(self, x, t):
        # t is ignored for compatibility
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.middle(self.pool(e3))
        d2 = self.up2(m)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        d0 = self.up0(d1)
        d0 = torch.cat([d0, e1], dim=1)
        d0 = self.dec0(d0)
        out = self.outc(d0)
        return out 