"""
Dual-Pyramid registration Network for registration
"""
import torch
import torch.nn as nn
from models.module import VGGBlock
from models.module import SpatialTransformer

class PRNet(nn.Module):
    def __init__(self, input_channels=1):
        super(PRNet, self).__init__()
        self.input_channels = input_channels
        nb_filter = [32, 64, 128, 256, 512]
        self.st_1 = SpatialTransformer((32, 32))
        self.st_2 = SpatialTransformer((64, 64))
        self.st_3 = SpatialTransformer((128, 128))
        self.st_4 = SpatialTransformer((256, 256))

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.head = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv0_0 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.conv_field_1 = nn.Sequential(nn.Conv2d(nb_filter[4] * 2, 2, kernel_size=(3, 3), padding=1),
                                          nn.BatchNorm2d(2),
                                          nn.ReLU())
        self.conv_field_2 = nn.Sequential(nn.Conv2d(nb_filter[3] * 2, 2, kernel_size=(3, 3), padding=1),
                                          nn.BatchNorm2d(2),
                                          nn.ReLU())
        self.conv_field_3 = nn.Sequential(nn.Conv2d(nb_filter[2] * 2, 2, kernel_size=(3, 3), padding=1),
                                          nn.BatchNorm2d(2),
                                          nn.ReLU())
        self.conv_field_4 = nn.Sequential(nn.Conv2d(nb_filter[1] * 2, 2, kernel_size=(3, 3), padding=1),
                                          nn.BatchNorm2d(2),
                                          nn.ReLU())
        self.final = nn.Conv2d(nb_filter[0] * 2, 2, kernel_size=(3, 3), padding=1)

    def forward(self, cbct, plct):
        """
        :param cbct: [n_sample, n_slice, w, h]
        :param plct: [n_sample, 1, w, h]
        :return: field [n_sample, 2, w, h]
        """
        cbct = self.head(cbct)
        x0_0 = self.conv0_0(cbct)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        plct = self.head(plct)
        x0_1 = self.conv0_0(plct)
        x1_1 = self.conv1_0(self.pool(x0_1))
        x2_1 = self.conv2_0(self.pool(x1_1))
        x3_1 = self.conv3_0(self.pool(x2_1))
        x4_1 = self.conv4_0(self.pool(x3_1))

        f3 = self.up(self.conv_field_1(torch.cat([x4_0, x4_1], 1)))
        u3_0 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        u3_1 = self.conv3_1(torch.cat([x3_1, self.up(x4_1)], 1))

        f2 = self.up(self.conv_field_2(torch.cat([self.st_1(u3_0, f3), u3_1], 1)))
        u2_0 = self.conv2_2(torch.cat([x2_0, self.up(u3_0)], 1))
        u2_1 = self.conv2_2(torch.cat([x2_1, self.up(u3_1)], 1))

        f1 = self.up(self.conv_field_3(torch.cat([self.st_2(u2_0, f2), u2_1], 1)))
        u1_0 = self.conv1_3(torch.cat([x1_0, self.up(u2_0)], 1))
        u1_1 = self.conv1_3(torch.cat([x1_1, self.up(u2_1)], 1))

        f0 = self.up(self.conv_field_4(torch.cat([self.st_3(u1_0, f1), u1_1], 1)))
        u0_0 = self.conv0_4(torch.cat([x0_0, self.up(u1_0)], 1))
        u0_1 = self.conv0_4(torch.cat([x0_1, self.up(u1_1)], 1))

        output = self.final(torch.cat([self.st_4(u0_0, f0), u0_1], 1))
        return output
