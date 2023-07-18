"""
Self-recovery Network
"""

import torch.nn as nn
from models.module import ResBlock,AEDown,AEUp

class resblock(nn.Module):
    def __init__(self, channels, num_layers=2):
        super(resblock, self).__init__()
        layers = [
            ResBlock(channels) for
        i in range(num_layers)]
        self.resblock = nn.Sequential(*layers)

    def forward (self, x):
        return self.resblock(x)

class ClassNet(nn.Module):
    def __init__(self):
        super(ClassNet, self).__init__()
        self.lay1_0 = AEDown(in_channels=1, out_channels=32, num_layers=1)
        self.lay1_1 = AEDown(in_channels=32, out_channels=32, num_layers=2)     # 8 128 128
        self.lay2_0 = AEDown(in_channels=32, out_channels=64, num_layers=1)
        self.lay2_1 = AEDown(in_channels=64, out_channels=64, num_layers=2)      # 16 64 64
        self.lay3_0 = AEDown(in_channels=64, out_channels=128, num_layers=1)
        self.lay3_1 = AEDown(in_channels=128, out_channels=128, num_layers=3)   # 32 64 64
        self.lay4_0 = AEDown(in_channels=128, out_channels=256, num_layers=1)
        self.lay4_1 = AEDown(in_channels=256, out_channels=256, num_layers=3)   # 32 64 64
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self._classifier = nn.Linear(256, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.lay1_0(x)
        x = self.lay1_1(x)
        x = self.maxpool(x)
        x = self.lay2_0(x)
        x = self.lay2_1(x)
        x = self.maxpool(x)
        x = self.lay3_0(x)
        x = self.lay3_1(x)
        x = self.maxpool(x)
        x = self.lay4_0(x)
        y = self.lay4_1(x)

        return y

class SelfNet(nn.Module):  # # final output channel number is 32
    def __init__(self):
        super(SelfNet, self).__init__()
        self.lay1_0 = AEDown(in_channels=1, out_channels=8, num_layers=1)
        self.lay1_1 = resblock(8,2)     # 8 128 128
        self.lay2_0 = AEDown(in_channels=8, out_channels=16, num_layers=1)
        self.lay2_1 = resblock(16,4)      # 16 64 64
        self.lay3_0 = AEDown(in_channels=16, out_channels=32, num_layers=1)
        self.lay3_1 = resblock(32,8)   # 32 32 32
        self.lay4_0 = AEDown(in_channels=32, out_channels=64, num_layers=1)
        self.lay4_1 = resblock(64,16)   # 64 32 32

        self.lay5 = resblock(64,16)   # 64 32 32
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2))
        self.lay6 = resblock(32,8)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2))
        self.lay7 = resblock(16, 4)
        self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=(2, 2), stride=(2, 2))
        self.lay8_1 = resblock(8, 2)
        self.lay8_0 = AEUp(in_channels=8, out_channels=1, num_layers=1)

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        x = self.lay1_0(x)
        x = self.lay1_1(x)
        x = self.maxpool(x)
        x = self.lay2_0(x)
        x = self.lay2_1(x)
        x = self.maxpool(x)
        x = self.lay3_0(x)
        x = self.lay3_1(x)
        x = self.maxpool(x)
        x = self.lay4_0(x)
        x = self.lay4_1(x)

        x = self.lay5(x)
        x = self.deconv1(x)
        x = self.lay6(x)
        x = self.deconv2(x)
        x = self.lay7(x)
        x = self.deconv3(x)
        x = self.lay8_1(x)
        out = self.lay8_0(x)

        return out

    def get_allfeatures(self,x):
        x = self.lay1_0(x)
        x = self.lay1_1(x)
        y1 = x
        x = self.maxpool(x)
        x = self.lay2_0(x)
        x = self.lay2_1(x)
        y2 = x
        x = self.maxpool(x)
        x = self.lay3_0(x)
        x = self.lay3_1(x)
        y3 = x
        x = self.maxpool(x)
        x = self.lay4_0(x)
        x = self.lay4_1(x)
        y4 = x
        return y1,y2,y3,y4

# class SelfNet64(nn.Module):  # final output channel number is 64
#     def __init__(self):
#         super(SelfNet64, self).__init__()
#         self.lay1_0 = AEDown(in_channels=1, out_channels=8, num_layers=1)
#         self.lay1_1 = resblock(8,2)     # 8 128 128
#         self.lay2_0 = AEDown(in_channels=8, out_channels=16, num_layers=1)
#         self.lay2_1 = resblock(16,4)      # 16 64 64
#         self.lay3_0 = AEDown(in_channels=16, out_channels=32, num_layers=1)
#         self.lay3_1 = resblock(32,8)   # 32 64 64
#
#         self.lay4 = resblock(32,8)   # 32 64 64
#         self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2))
#         self.lay5 = resblock(16, 4)
#         self.deconv2 = nn.ConvTranspose2d(16, 8, kernel_size=(2, 2), stride=(2, 2))
#         self.lay6_1 = resblock(8, 2)
#         self.lay6_0 = AEUp(in_channels=8, out_channels=1, num_layers=1)
#
#         self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
#
#     def forward(self, x):
#         x = self.lay1_0(x)
#         x = self.lay1_1(x)
#         x = self.maxpool(x)
#         x = self.lay2_0(x)
#         x = self.lay2_1(x)
#         x = self.maxpool(x)
#         x = self.lay3_0(x)
#         x = self.lay3_1(x)
#
#         x = self.lay4(x)
#         x = self.deconv1(x)
#         x = self.lay5(x)
#         x = self.deconv2(x)
#         x = self.lay6_1(x)
#         out = self.lay6_0(x)
#
#         return out
#
#     def get_last_shared_layer(self,x):
#         y = self.lay1_0(x)
#         return y
#
#     def get_allfeatures(self,x):
#         x = self.lay1_0(x)
#         x = self.lay1_1(x)
#         y1 = x
#         x = self.maxpool(x)
#         x = self.lay2_0(x)
#         x = self.lay2_1(x)
#         y2 = x
#         x = self.maxpool(x)
#         x = self.lay3_0(x)
#         x = self.lay3_1(x)
#         y3 = x
#         return y1,y2,y3