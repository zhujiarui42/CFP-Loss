"""
MTFSnet
"""

from models.module import Conv,ResBlock,VGGBlock
import torch
import torch.nn as nn
from models.module import AEDown,AEUp,SpatialTransformer

class resblock(nn.Module):
    def __init__(self, channels, num_layers=2):
        super(resblock, self).__init__()
        layers = [
            ResBlock(channels) for
        i in range(num_layers)]
        self.resblock = nn.Sequential(*layers)

    def forward (self, x):
        return self.resblock(x)


class Encoder(nn.Module):
    def __init__(self, ch=[1, 8, 16, 32, 64, 64], n=[2, 4, 8, 16]):
        super(Encoder, self).__init__()

        layers = [self.AElayer(ch[i], ch[i + 1], n[i]) for i in range(n)]

        self.layers = nn.Sequential(*layers)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def AElayer(inch, outch, num_layers):
        return AEDown(in_channels=inch, out_channels=outch, num_layers=1), resblock(outch, num_layers)

    def forward(self, x):
        x_set = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if (i + 1) % 2 == 0:
                x_set.append(x)
                x = self.maxpool(x)
        return x, x_set

    def get_allfeatures(self, x):
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
        return y1, y2, y3, y4

class Decoder_self(nn.Module):
    def __init__(self, ch=[64, 32, 16, 8, 1], n=[16, 8, 4, 2]):
        super(Decoder_self, self).__init__()
        layers = []
        for i in range(len(n)):
            layers.append(resblock(ch[i], n[i]))
            layers.append(nn.ConvTranspose2d(ch[i], ch[i+1], kernel_size=(2, 2), stride=(2, 2)))
        layers[-1] = AEUp(in_channels=ch[-2], out_channels=ch[-1], num_layers=1)
        self.layers = nn.Sequential(*layers)

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Classifier(nn.Module):
    def __init__(self,in_channels):
        super(Classifier, self).__init__()
        self._classifier = nn.Linear(in_channels, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self,x,y):
        """
        :param x: [B, 256, 32, 32]
        :return:
        """
        B = x.size(0)
        x = self.gap(x).reshape(B,-1)
        y = self.gap(y).reshape(B,-1)
        x1 = self.fcl(x).softmax(dim=-1)
        y1 = self.fcl(y).softmax(dim=-1)
        prob = torch.cat([x1,y1],dim=0)

        return prob

class Decoder_dvf:
    def __init__(self, input_channels=1):
        super(Decoder_dvf, self).__init__()

        self.input_channels = input_channels
        nb_filter = [32, 64, 128, 256]
        self.st_1 = SpatialTransformer((32, 32))
        self.st_2 = SpatialTransformer((64, 64))
        self.st_3 = SpatialTransformer((128, 128))
        self.st_4 = SpatialTransformer((256, 256))

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

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

    def forward(self, x, y):
        x0_0, x1_0, x2_0, x3_0, x4_0 = x
        x0_1, x1_1, x2_1, x3_1, x4_1 = y

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

class MTFSNet(nn.Module):
    def __init__(self, ):
        super(MTFSNet, self).__init__()
        #define shared encoder and an individual encoder for registration task
        self.encoder_cbct = Encoder()
        self.encoder_plct = Encoder()
        #define classifier
        self.fcl = nn.Linear(in_features=32*32*64,out_features=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        #define decoders for three sub-tasks
        self.decoder_plct = Decoder_self()
        self.decoder_dvf = Decoder_dvf()
        self.decoder_class = Classifier()
        #define individual weights for gradnorm calculation
        self.weight = torch.nn.Parameter(torch.ones(3).float())

    def forward(self,cbct,plct):
        cbct_feature = self.encoder_cbct.forward(cbct)[0]
        plct_feature = self.encoder_plct.forward(plct)[0]
        cbctset = self.encoder_cbct.forward(cbct)[1]
        plctset = self.encoder_plct.forward(plct)[1]
        plct_re = self.decoder_plct.forward(plct_feature)
        dvf = self.decoder_dvf.forward(cbctset,plctset)
        prob = self.decoder_class.forward(cbct_feature,plct_feature)

        return plct_re,dvf,prob

    def get_features(self,cbct,plct):
        x1,x2,x3,x4 = self.encoder_cbct.get_allfeatures(cbct)
        y1,y2,y3,y4 = self.encoder_plct.get_allfeatures(plct)

        return x1,x2,x3,x4,y1,y2,y3,y4

