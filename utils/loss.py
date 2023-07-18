"""
GAN Loss and Multi Task Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import SpatialTransformer
from utils.metrics import SSIMLoss
from models.MultitaskModels.MTFSnet import MTFSNet


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.

    This class is adopted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        noise = torch.rand(prediction.size()).cuda()
        if target_is_real:
            target_tensor = 1 - noise * 0.1
        else:
            target_tensor = noise * 0.1
        return target_tensor.detach()  # .expand_as(prediction)

    def _forward(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

    def forward_d(self, pred_fake, pred_real):
        """ Get the loss that updates the discriminator
        fake is detach
        """

        loss_fake = self._forward(pred_fake.detach(), False)
        loss_real = self._forward(pred_real, True)

        loss = (loss_real + loss_fake) * 0.5
        return loss

    def forward_g(self, pred_fake):
        """
        not detach
        """
        loss = self._forward(pred_fake, True)

        return loss

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def __call__(self, y_pred):
        dx = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dy = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class GenLoss(nn.Module):
    def __init__(self):
        super(GenLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.grad = Grad()
        self.st = SpatialTransformer((256, 256, 256))

    def forward(self, lpuq, en_plct, field=None, cbct=None, plct=None):
        # field = field.unsqueeze(-1).repeat(1, 1, 1, 1, 256)
        # cbct_warp = self.st(cbct, field)
        en_loss = self.l1(lpuq, en_plct)
        # reg_loss = self.l2(cbct_warp, plct)
        # grad = self.grad(field)
        return en_loss  # , reg_loss, grad

class MultiTaskLoss(object):
    def __init__(self):
        self.d_loss = GANLoss().cuda()
        self.g_loss = GenLoss().cuda()

class CycleGANLoss(nn.Module):
    def __init__(self):
        super(CycleGANLoss, self).__init__()
        self.cycle_loss = MixLoss()
        self.identity_loss = MixLoss()
        self.adversarial_loss = torch.nn.MSELoss()

class MixLoss(nn.Module):
    """
    copy from https://github.com/lizhengwei1992/MS_SSIM_pytorch
    """
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range=1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0,
                 cuda_dev=0, ):
        super(MixLoss, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3 * len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3 * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, pred, target):
        b, c, h, w = pred.shape
        mux = F.conv2d(pred, self.g_masks, groups=1, padding=self.pad)
        muy = F.conv2d(target, self.g_masks, groups=1, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(pred * pred, self.g_masks, groups=1, padding=self.pad) - mux2
        sigmay2 = F.conv2d(target * target, self.g_masks, groups=1, padding=self.pad) - muy2
        sigmaxy = F.conv2d(pred * target, self.g_masks, groups=1, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(pred, target, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=1, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()

class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()
        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.grad = Grad()

    def forward(self, x, y, field):
        l1 = self.l1(x, y)
        grad = self.grad(field)
        return l1 + grad * 0.01

class AAPMNetLoss(nn.Module):
    def __init__(self):
        super(AAPMNetLoss, self).__init__()
        self.dn_loss = SSIMLoss()
        self.mul_loss = nn.MSELoss()
        self.trans = MixLoss()

    def forward(self, dn_cbct, trans_cbct, warp_cbct, plct, filed):
        # dn_loss = self.dn_loss(dn_cbct, plct)
        mul_loss = self.mul_loss(warp_cbct, plct)
        trans_loss = self.trans(trans_cbct, plct)
        return mul_loss + trans_loss  # + dn_loss

class CFPLoss(nn.Module):
    def __init__(self):
        super(CFPLoss, self).__init__()
        self.l2 = nn.MSELoss()
        self.lossnet = MTFSNet()
        self.lossnet.load_state_dict(torch.load(f"output/mtfsnet/model.npy")['state_dict'])

    def forward(self, x, y):
        #get features from the encoder
        y11,y12,y13,y14 = self.lossnet.get_allfeatures(x)
        y21,y22,y23,y24 = self.lossnet.get_allfeatures(y)
        #flatten the feature maps(tensors) along channel axis
        y11f = torch.flatten(y11, start_dim=1)
        y12f = torch.flatten(y12, start_dim=1)
        y13f = torch.flatten(y13, start_dim=1)
        y14f = torch.flatten(y14, start_dim=1)
        y21f = torch.flatten(y21, start_dim=1)
        y22f = torch.flatten(y22, start_dim=1)
        y23f = torch.flatten(y23, start_dim=1)
        y24f = torch.flatten(y24, start_dim=1)
        #calculate inner vector product
        vp11 = torch.dot(y11f,y11f.t)
        vp12 = torch.dot(y12f,y12f.t)
        vp13 = torch.dot(y13f,y13f.t)
        vp14 = torch.dot(y14f,y14f.t)
        vp21 = torch.dot(y21f,y21f.t)
        vp22 = torch.dot(y22f,y22f.t)
        vp23 = torch.dot(y23f,y23f.t)
        vp24 = torch.dot(y24f,y24f.t)
        #calculate Euclidean distance
        loss_content1 = self.l2(y11,y21)
        loss_content2 = self.l2(y12,y22)
        loss_content3 = self.l2(y13,y23)
        loss_content4 = self.l2(y14,y24)
        loss_style1 = self.l2(vp11,vp21)
        loss_style2 = self.l2(vp12,vp22)
        loss_style3 = self.l2(vp13,vp23)
        loss_style4 = self.l2(vp14,vp24)

        # the final loss fucntion can be customized accordling to the training status and disired traits
        loss = 0.4 * loss_content1 + 0.4* loss_content2 + (0.5 * loss_style1+loss_style2+loss_style3+loss_style4)

        return loss

class GradNormLoss(nn.Module):
    def __init__(self):
        super(GradNormLoss, self).__init__()
        self.mix = MixLoss()
        self.reg = RegLoss()
        self.ce = nn.CrossEntropyLoss()
        self.weights = nn.Parameter(torch.ones(7).float())

    def forward(self, x, y):
        loss = []
        loss += [self.mix(x['cbct'], y['cbct'])]
        loss += [self.mix(x['plct'], y['plct'])]
        loss += [self.mix(x['moving_cbct'], y['moving_cbct'])]
        loss += [self.mix(x['moving_plct'], y['moving_plct'])]
        loss += [self.reg(x['warp_cbct'], y['cbct'], x['dvf_cbct'])]
        loss += [self.reg(x['warp_plct'], y['plct'], x['dvf_plct'])]
        loss += [self.ce(x['prob'], y['label'])]
        loss = torch.stack(loss)
        return loss