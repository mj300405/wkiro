import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        _, _, kH, kW = self.weight.size()
        yc, xc = kH // 2, kW // 2
        self.mask[:, :, yc+1:, :] = 0
        self.mask[:, :, yc, xc+1:] = 0
        if mask_type == 'A':
            self.mask[:, :, yc, xc] = 0
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class PixelCNN(nn.Module):
    def __init__(self, num_embeddings, input_shape=(7, 7), n_channels=64, n_layers=7):
        super().__init__()
        self.input_shape = input_shape
        self.embed = nn.Embedding(num_embeddings, n_channels)
        self.conv1 = MaskedConv2d('A', n_channels, n_channels, 7, padding=3)
        self.convs = nn.ModuleList([
            MaskedConv2d('B', n_channels, n_channels, 7, padding=3) for _ in range(n_layers)
        ])
        self.conv_out = nn.Conv2d(n_channels, num_embeddings, 1)
    def forward(self, x):
        # x: (B, H, W) integer indices
        x = self.embed(x)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.conv1(x)
        x = F.relu(x)
        for conv in self.convs:
            x = F.relu(conv(x))
        x = self.conv_out(x)
        return x  # logits (B, num_embeddings, H, W) 