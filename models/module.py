"""
modules for models
"""
import torch
from torch import nn
from torch.nn import functional as F

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3, 3), stride=(1, 1), padding=1,
                 norm='none', act='none', dropout=None):
        super(Conv, self).__init__()
        layer = [nn.Conv2d(in_channels, out_channels, kernel, stride, padding)]

        if norm == 'batch':
            layer += [nn.BatchNorm2d(out_channels)]
        elif norm == 'instance':
            layer += [nn.InstanceNorm2d(out_channels)]
        elif norm == 'layer':
            layer += [LayerNorm(out_channels)]

        if act == 'relu':
            layer += [nn.ReLU(inplace=True)]
        elif act == 'sigmoid':
            layer += [nn.Sigmoid()]
        elif act == 'lrelu':
            layer += [nn.LeakyReLU(0.2, inplace=True)]
        elif act == 'tanh':
            layer += [nn.Tanh()]
        elif act == 'gelu':
            layer += [nn.GELU()]
        elif act == 'selu':
            layer += [nn.SELU(inplace=True)]

        if dropout is not None:
            layer += [nn.Dropout(dropout)]

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            Conv(in_channels, middle_channels, norm='batch', act='relu', dropout=0.1),
            Conv(middle_channels, out_channels, norm='batch', act='relu', dropout=0.1)
        )

    def forward(self, x):
        out = self.block(x)
        return out

class ResBlock(nn.Module):
    def __init__(self, channel, norm='batch', act='relu', dropout=0.1):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            Conv(channel, channel, (3, 3), (1, 1), 1, norm, act, dropout),
            Conv(channel, channel, (3, 3), (1, 1), 1, norm, act, dropout),
        )

    def forward(self, x): return self.res_block(x) + x

# "Squeeze and Excitation" SE-Block

class SqEx(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y

# Residual block using SE-Block

class ResBlockSE(nn.Module):

    def __init__(self, n_features):
        super(ResBlockSE, self).__init__()

        self.norm1 = nn.BatchNorm2d(n_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm2 = nn.BatchNorm2d(n_features)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.sqex = SqEx(n_features)

    def forward(self, x):

        y = self.conv1(self.relu1(self.norm1(x)))
        y = self.conv2(self.relu2(self.norm2(y)))

        y = self.sqex(y)

        y = torch.add(x, y)

        return y

#modules for autoencoder

class AEDown(nn.Module):
    """
    input:N*C*D*H*W
    """

    def __init__(self, in_channels, out_channels, num_layers=2):
        super(AEDown, self).__init__()
        layers = [
            Conv(in_channels, out_channels, norm='batch', act='relu') if i == 0 else Conv(out_channels, out_channels,
                                                                                          norm='batch', act='relu') for
            i in range(num_layers)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class AEUp(nn.Module):
    """
    input:N*C*D*H*W
    """

    def __init__(self, in_channels, out_channels, num_layers=2):
        super(AEUp, self).__init__()
        layers = [
            Conv(in_channels, out_channels, norm='batch', act='relu') if i == 0 else Conv(out_channels, out_channels,
                                                                                          norm='batch', act='relu') for
            i in range(num_layers)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = F.layer_norm(x, x.shape[1:], eps=self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

# modules for warpst

class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class ResizeTransform(nn.Module):

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x