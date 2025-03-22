import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvResidual(nn.Module):
    """
    Double convolution block with residual connection for U-Net
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_p=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        # Main path
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection if input and output channels differ
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.double_conv(x)
        return x + residual  # Add residual connection


class ChannelAttention(nn.Module):
    """
    Channel attention module to focus on important feature channels
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on important spatial locations
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate channel-wise attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        return x * attention


class AttentionBlock(nn.Module):
    """
    Attention block combining channel and spatial attention
    """
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv with attention
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvResidual(in_channels, out_channels, dropout_p=dropout_p),
            AttentionBlock(out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv with attention
    """
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_p=0.1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
                DoubleConvResidual(in_channels, out_channels, in_channels // 2, dropout_p=dropout_p),
                AttentionBlock(out_channels)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                DoubleConvResidual(in_channels, out_channels, dropout_p=dropout_p),
                AttentionBlock(out_channels)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final convolution layer with dropout for regularization
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.dropout(x)
        return self.conv(x)


class UNetEnhanced(nn.Module):
    """
    Enhanced U-Net architecture with attention and residual connections
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, dropout_p=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout_p = dropout_p

        # Encoder (downsampling path) with residual blocks and attention
        self.inc = nn.Sequential(
            DoubleConvResidual(n_channels, 64, dropout_p=dropout_p),
            AttentionBlock(64)
        )
        self.down1 = Down(64, 128, dropout_p=dropout_p)
        self.down2 = Down(128, 256, dropout_p=dropout_p)
        self.down3 = Down(256, 512, dropout_p=dropout_p)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, dropout_p=dropout_p)
        
        # Decoder (upsampling path) with attention
        self.up1 = Up(1024, 512 // factor, bilinear, dropout_p=dropout_p)
        self.up2 = Up(512, 256 // factor, bilinear, dropout_p=dropout_p)
        self.up3 = Up(256, 128 // factor, bilinear, dropout_p=dropout_p)
        self.up4 = Up(128, 64, bilinear, dropout_p=dropout_p)
        
        # Final output layer
        self.outc = OutConv(64, n_classes, dropout_p=dropout_p)

    def forward(self, x):
        # Encoder path with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output layer (no activation - will be handled in loss function)
        logits = self.outc(x)
        return logits


# Factory function to create the model
def create_model(n_channels=3, n_classes=1, bilinear=True, dropout_p=0.1):
    model = UNetEnhanced(n_channels, n_classes, bilinear, dropout_p)
    return model 