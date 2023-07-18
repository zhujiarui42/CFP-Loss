"""
Metrics for model
"""
import warnings
import numpy as np
import math
import torch
from torch.nn import functional as F

class Scorer(object):
    """

    """
    def __init__(self):
        self.logger = None
        self.data_list = None
        self.metrics = [
                ("psnr", psnr),
                ("mae", mae),
                ("ssim", ssim),
                ("ncc", ncc),
                ("vif", vif)
            ]
        self.metricscs = [
        ("acc", acc)  # for classification
        ]


    def cal(self, x, y , isclass = False):
        """
        :param x: dict
        :param y: dict
        :return:
        """
        score = {}
        if isinstance(x, dict) is False:
            if isclass is True:
                for method, scorer in self.metricscs:
                    m = scorer(x, y).item()
                    score[f'base_{method}'] = m
                    self.logger[f'base_{method}'] += m
            else:
                for method, scorer in self.metrics:
                    m = scorer(x, y).item()
                    score[f'base_{method}'] = m
                    self.logger[f'base_{method}'] += m
        else:
            for i in x.keys():
                for method, scorer in self.metrics:
                    m = scorer(x[i], y[i]).item()
                    score[f'{i}_{method}'] = m
                    self.logger[f'{i}_{method}'] += m
                    self.logger[f'mean_{method}'] += m
            for method, scorer in self.metrics:
                # self.logger[f'mean_{method}'] /= len(self.data_list)
                self.logger[f'mean_{method}'] /= 4
        return score

    def score_log(self, data_list=None , isclass = False):
        if data_list is not None:
            self.data_list = data_list
            self.logger = {}
            for data in data_list:
                for method, scorer in self.metrics:
                    self.logger[f'{data}_{method}'] = 0.
            for method,scorer in self.metrics:
                self.logger[f'mean_{method}'] = 0.
        else:
            self.logger = {}
            if isclass is True:
                for method, scorer in self.metrics:
                    self.logger[f'base_{method}'] = 0.
                for method, scorer in self.metricscs:
                    self.logger[f'base_{method}'] = 0.
            else:
                for method, scorer in self.metrics:
                    self.logger[f'base_{method}'] = 0.

    def mean_score(self, n):
        for i in self.logger.keys():
            self.logger[i] /= n
        return self.logger

def psnr(pre, target, max_value=1.):
    mse = torch.mean((pre - target) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = torch.tensor(max_value)
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def acc(pred,label):
    pred = pred[:, 1]
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0

    return 100 * (pred == label).float().mean()


def mae(pre, target):
    return torch.mean(torch.abs(pre - target))

def eudispix(pre, target):
    I = (pre - target)**2
    s = torch.sum(I,dim=[1,2,3])
    sq = torch.sqrt(s)
    m = torch.mean(sq)
    return m

def eudischannel(pre, target):
    I = (pre - target)**2
    s = torch.sum(I,dim=1)
    sq = torch.sqrt(s)
    m = torch.mean(sq)
    return m

def ncc(pre, target, win=None):
    """
    Local (over window) normalized cross correlation loss.
    """
    I = target
    J = pre

    # get dimension of volume
    # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    win = [9] * ndims if win is None else win

    # compute filters
    sum_filt = torch.ones([1, 1, *win])

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)

    # compute CC squares
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + 1e-5)

    return torch.mean(cc)


def vif(pre, target):
    sigma_nsq = 2
    eps = 1e-10

    num = 0.0
    den = 0.0

    pre = pre * 255
    target = target * 255

    for scale in range(1, 5):

        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        win = _fspecial_gauss_1d(11, sd)
        win = win.repeat([pre.shape[1]] + [1] * (len(pre.shape) - 1))

        if (scale > 1):
            pre = gaussian_filter(pre, win)
            target = gaussian_filter(target, win)
            pre = pre[:, :, ::2, ::2]
            target = target[:, :, ::2, ::2]

        mu1 = gaussian_filter(pre, win)
        mu2 = gaussian_filter(target, win)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = gaussian_filter(pre * pre, win) - mu1_sq
        sigma2_sq = gaussian_filter(target * target, win) - mu2_sq
        sigma12 = gaussian_filter(pre * target, win) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += torch.sum(torch.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += torch.sum(torch.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den

    if torch.isnan(vifp):
        return torch.tensor(0.)
    else:
        return vifp


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(X, Y, data_range=1., size_average=True, win_size=11, win_sigma=1.5, win=None, K=(0.01, 0.03),
         nonnegative_ssim=False):
    """ interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(X, Y, data_range=1., size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None,
            K=(0.01, 0.03)):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
            2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights).to(X.device, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=1., size_average=True, channel=1):
        super(SSIMLoss, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, X, Y):
        return 1-ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average)


class MS_SSIMLoss(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=1., size_average=True, channel=1, weights=None):
        super(MS_SSIMLoss, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights

    def forward(self, X, Y):
        return 1-ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range,
                       weights=self.weights)
