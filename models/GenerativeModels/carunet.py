"""
unet,resunet and channel attention resunet (ResunetSE)models
"""
import torch
import torch.nn as nn
from models.module import ResBlock,ResBlockSE,VGGBlock, Conv

class UNet(nn.Module):
    """
    UNet with vgg block
    """
    def __init__(self):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.dn = UNet()
        # self.dn.load_state_dict(torch.load(f"output/unet/model.npy")['state_dict'])

        self.conv0_0 = VGGBlock(1, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, x):
        # x = self.dn(x)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output ,  x4_0 # .sigmoid()

class ResUNet(nn.Module):
    """
    UNet with Res Block
    """
    def __init__(self, args=None):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = Conv(1, nb_filter[0], (7, 7), (1, 1), 3, 'batch', 'relu')
        self.conv0_1 = Conv(nb_filter[0], nb_filter[1], (3, 3), (1, 1), 1, 'batch', 'relu')
        self.conv1_2 = Conv(nb_filter[1], nb_filter[2], (3, 3), (1, 1), 1, 'batch', 'relu')
        self.conv2_3 = Conv(nb_filter[2], nb_filter[3], (3, 3), (1, 1), 1, 'batch', 'relu')
        self.conv3_4 = Conv(nb_filter[3], nb_filter[4], (3, 3), (1, 1), 1, 'batch', 'relu')
        self.conv0 = ResBlock(nb_filter[0], 'batch')
        self.conv1 = ResBlock(nb_filter[1], 'batch')
        self.conv2 = ResBlock(nb_filter[2], 'batch')
        self.conv3 = ResBlock(nb_filter[3], 'batch')
        self.conv4 = ResBlock(nb_filter[4], 'batch')

        self.conv4_3 = Conv(nb_filter[4] + nb_filter[3], nb_filter[3], (3, 3), (1, 1), 1, 'batch', 'lrelu')
        self.conv3_2 = Conv(nb_filter[3] + nb_filter[2], nb_filter[2], (3, 3), (1, 1), 1, 'batch', 'lrelu')
        self.conv2_1 = Conv(nb_filter[2] + nb_filter[1], nb_filter[1], (3, 3), (1, 1), 1, 'batch', 'lrelu')
        self.conv1_0 = Conv(nb_filter[1] + nb_filter[0], nb_filter[0], (3, 3), (1, 1), 1, 'batch', 'lrelu')
        self.conv3_up = ResBlock(nb_filter[3], 'batch', 'lrelu')
        self.conv2_up = ResBlock(nb_filter[2], 'batch', 'lrelu')
        self.conv1_up = ResBlock(nb_filter[1], 'batch', 'lrelu')
        self.conv0_up = ResBlock(nb_filter[0], 'batch', 'lrelu')

        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)  # [B, 8, 256, ...]
        x0 = self.conv0(x0_0)  # [B, 8, 256, ...]

        x0_1 = self.conv0_1(self.pool(x0))  # [B, 16, 128, ...]
        x1 = self.conv1(x0_1)  # [B, 16, 128, ...]

        x1_2 = self.conv1_2(self.pool(x1))  # [B, 32, 64, ...]
        x2 = self.conv2(x1_2)  # [B, 32, 64, ...]

        x2_3 = self.conv2_3(self.pool(x2))  # [B, 64, 32, ...]
        x3 = self.conv3(x2_3)  # [B, 64, 32, ...]

        x3_4 = self.conv3_4(self.pool(x3))  # [B, 128, 16, ...]
        x4 = self.conv4(x3_4)  # [B, 128, 16, ...]

        x4_3 = self.conv4_3(torch.cat([x3, self.up(x4)], 1))  # [B, 64, 32, ...]
        x3_up = self.conv3_up(x4_3)

        x3_2 = self.conv3_2(torch.cat([x2, self.up(x3_up)], 1))  # [B, 32, 64, ...]
        x2_up = self.conv2_up(x3_2)

        x2_1 = self.conv2_1(torch.cat([x1, self.up(x2_up)], 1))  # [B, 16, 128, ...]
        x1_up = self.conv1_up(x2_1)

        x1_0 = self.conv1_0(torch.cat([x0, self.up(x1_up)], 1))  # [B, 8, 256, ...]
        x0_up = self.conv0_up(x1_0)

        output = self.final(x0_up)
        return output.sigmoid()


class CARUNet(nn.Module):
    """
    UNet with Res Block & SE channel attention blocks
    """
    def __init__(self, args=None):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.dn = UNet()
        # self.dn.load_state_dict(torch.load(f"output/unet/model.npy")['state_dict'])

        self.conv0_0 = Conv(1, nb_filter[0], (7, 7), (1, 1), 3, 'batch', 'relu')
        self.conv0_1 = Conv(nb_filter[0], nb_filter[1], (3, 3), (1, 1), 1, 'batch', 'relu')
        self.conv1_2 = Conv(nb_filter[1], nb_filter[2], (3, 3), (1, 1), 1, 'batch', 'relu')
        self.conv2_3 = Conv(nb_filter[2], nb_filter[3], (3, 3), (1, 1), 1, 'batch', 'relu')
        self.conv3_4 = Conv(nb_filter[3], nb_filter[4], (3, 3), (1, 1), 1, 'batch', 'relu')
        self.conv0 = ResBlockSE(nb_filter[0], 'batch')
        self.conv1 = ResBlockSE(nb_filter[1], 'batch')
        self.conv2 = ResBlockSE(nb_filter[2], 'batch')
        self.conv3 = ResBlockSE(nb_filter[3], 'batch')
        self.conv4 = ResBlockSE(nb_filter[4], 'batch')

        self.conv4_3 = Conv(nb_filter[4] + nb_filter[3], nb_filter[3], (3, 3), (1, 1), 1, 'batch', 'lrelu')
        self.conv3_2 = Conv(nb_filter[3] + nb_filter[2], nb_filter[2], (3, 3), (1, 1), 1, 'batch', 'lrelu')
        self.conv2_1 = Conv(nb_filter[2] + nb_filter[1], nb_filter[1], (3, 3), (1, 1), 1, 'batch', 'lrelu')
        self.conv1_0 = Conv(nb_filter[1] + nb_filter[0], nb_filter[0], (3, 3), (1, 1), 1, 'batch', 'lrelu')
        self.conv3_up = ResBlockSE(nb_filter[3], 'batch', 'lrelu')
        self.conv2_up = ResBlockSE(nb_filter[2], 'batch', 'lrelu')
        self.conv1_up = ResBlockSE(nb_filter[1], 'batch', 'lrelu')
        self.conv0_up = ResBlockSE(nb_filter[0], 'batch', 'lrelu')

        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, x):
        # x = self.dn(x)
        x0_0 = self.conv0_0(x)  # [B, 8, 256, ...]
        x0 = self.conv0(x0_0)  # [B, 8, 256, ...]

        x0_1 = self.conv0_1(self.pool(x0))  # [B, 16, 128, ...]
        x1 = self.conv1(x0_1)  # [B, 16, 128, ...]

        x1_2 = self.conv1_2(self.pool(x1))  # [B, 32, 64, ...]
        x2 = self.conv2(x1_2)  # [B, 32, 64, ...]

        x2_3 = self.conv2_3(self.pool(x2))  # [B, 64, 32, ...]
        x3 = self.conv3(x2_3)  # [B, 64, 32, ...]

        x3_4 = self.conv3_4(self.pool(x3))  # [B, 128, 16, ...]
        x4 = self.conv4(x3_4)  # [B, 128, 16, ...]

        x4_3 = self.conv4_3(torch.cat([x3, self.up(x4)], 1))  # [B, 64, 32, ...]
        x3_up = self.conv3_up(x4_3)

        x3_2 = self.conv3_2(torch.cat([x2, self.up(x3_up)], 1))  # [B, 32, 64, ...]
        x2_up = self.conv2_up(x3_2)

        x2_1 = self.conv2_1(torch.cat([x1, self.up(x2_up)], 1))  # [B, 16, 128, ...]
        x1_up = self.conv1_up(x2_1)

        x1_0 = self.conv1_0(torch.cat([x0, self.up(x1_up)], 1))  # [B, 8, 256, ...]
        x0_up = self.conv0_up(x1_0)

        output = self.final(x0_up)
        return output.sigmoid()
