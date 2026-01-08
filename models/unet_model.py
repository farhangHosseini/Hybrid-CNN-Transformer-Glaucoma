import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    Standard U-Net implementation for Iris Segmentation.
    """
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        
        self.u4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        
        self.up4 = DoubleConv(1024, 512)
        self.up3 = DoubleConv(512, 256)
        self.up2 = DoubleConv(256, 128)
        self.up1 = DoubleConv(128, 64)
        
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        bn = self.bottleneck(self.pool(c4))
        
        u4 = self.up4(torch.cat([self.u4(bn), c4], dim=1))
        u3 = self.up3(torch.cat([self.u3(u4), c3], dim=1))
        u2 = self.up2(torch.cat([self.u2(u3), c2], dim=1))
        u1 = self.up1(torch.cat([self.u1(u2), c1], dim=1))
        
        return self.out(u1)