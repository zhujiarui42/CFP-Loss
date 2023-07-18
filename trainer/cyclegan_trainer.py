import os

from trainer import BaseTrainer
from models.GenerativeModels.cyclegan import ReplayBuffer
import torch


class CycleGANTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(CycleGANTrainer, self).__init__(**kwargs)
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def train_one_epoch(self, ep):
        self.model.train()
        print("-"*10, f" {ep} ", "-"*10)
        log_loss = torch.zeros([7])
        for (_index, (plct, cbct)) in enumerate(self.train_loader):
            # get batch size data
            B, C, H, W = cbct.size()
            device = cbct.device
            # real data label is 1, fake data label is 0.
            real_label = torch.full((B, 1), 1, device=device, dtype=torch.float32)
            fake_label = torch.full((B, 1), 0, device=device, dtype=torch.float32)
            real_image_A = cbct
            real_image_B = plct
            ##############################################
            # (1) Update G network: Generators A2B and B2A
            ##############################################

            # Set G_A and G_B's gradients to zero
            self.optim['g'].zero_grad()

            # Identity loss
            # G_B2A(A) should equal A if real A is fed
            identity_image_A = self.model.netG_B2A(real_image_A)
            loss_identity_A = self.criterion.identity_loss(identity_image_A, real_image_A) * 5.0
            # G_A2B(B) should equal B if real B is fed
            identity_image_B = self.model.netG_A2B(real_image_B)
            loss_identity_B = self.criterion.identity_loss(identity_image_B, real_image_B) * 5.0

            # GAN loss
            # GAN loss D_A(G_A(A))
            fake_image_A = self.model.netG_B2A(real_image_B)
            fake_output_A = self.model.netD_A(fake_image_A)
            loss_GAN_B2A = self.criterion.adversarial_loss(fake_output_A, real_label)
            # GAN loss D_B(G_B(B))
            fake_image_B = self.model.netG_A2B(real_image_A)
            fake_output_B = self.model.netD_B(fake_image_B)
            loss_GAN_A2B = self.criterion.adversarial_loss(fake_output_B, real_label)

            # Cycle loss
            recovered_image_A = self.model.netG_B2A(fake_image_B)
            loss_cycle_ABA = self.criterion.cycle_loss(recovered_image_A, real_image_A) * 10.0

            recovered_image_B = self.model.netG_A2B(fake_image_A)
            loss_cycle_BAB = self.criterion.cycle_loss(recovered_image_B, real_image_B) * 10.0

            # Combined loss and calculate gradients
            errG = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            # Calculate gradients for G_A and G_B
            errG.backward()
            # Update G_A and G_B's weights
            self.optim['g'].step()

            ##############################################
            # (2) Update D network: Discriminator A
            ##############################################

            # Set D_A gradients to zero
            self.optim['d_a'].zero_grad()

            # Real A image loss
            real_output_A = self.model.netD_A(real_image_A)
            errD_real_A = self.criterion.adversarial_loss(real_output_A, real_label)

            # Fake A image loss
            fake_image_A = self.fake_A_buffer.push_and_pop(fake_image_A)
            fake_output_A = self.model.netD_A(fake_image_A.detach())
            errD_fake_A = self.criterion.adversarial_loss(fake_output_A, fake_label)

            # Combined loss and calculate gradients
            errD_A = (errD_real_A + errD_fake_A) / 2

            # Calculate gradients for D_A
            errD_A.backward()
            # Update D_A weights
            self.optim['d_a'].step()

            ##############################################
            # (3) Update D network: Discriminator B
            ##############################################

            # Set D_B gradients to zero
            self.optim['d_b'].zero_grad()

            # Real B image loss
            real_output_B = self.model.netD_B(real_image_B)
            errD_real_B = self.criterion.adversarial_loss(real_output_B, real_label)

            # Fake B image loss
            fake_image_B = self.fake_B_buffer.push_and_pop(fake_image_B)
            fake_output_B = self.model.netD_B(fake_image_B.detach())
            errD_fake_B = self.criterion.adversarial_loss(fake_output_B, fake_label)

            # Combined loss and calculate gradients
            errD_B = (errD_real_B + errD_fake_B) / 2

            # Calculate gradients for D_B
            errD_B.backward()
            # Update D_B weights
            self.optim['d_b'].step()

            d_loss = (errD_A + errD_B).item()
            d_loss_a = errD_A.item()
            d_loss_b = errD_B.item()
            g_loss = errG.item()
            loss_identity = (loss_identity_A + loss_identity_B).item()
            loss_gan = (loss_GAN_A2B + loss_GAN_B2A).item()
            loss_cycle = (loss_cycle_ABA + loss_cycle_BAB).item()

            train_info = f"\r{_index:d} " + \
                         f"Loss_D: {d_loss:.4f} " + \
                         f"Loss_D_A: {d_loss_a:.4f} " + \
                         f"Loss_D_B: {d_loss_b:.4f} " + \
                         f"Loss_G: {g_loss:.4f} " + \
                         f"Loss_G_identity: {loss_identity:.4f} " + \
                         f"loss_G_GAN: {loss_gan:.4f} " + \
                         f"loss_G_cycle: {loss_cycle:.4f}"

            _loss = torch.tensor([d_loss, g_loss, d_loss_a, d_loss_b, loss_identity, loss_gan, loss_cycle]).detach()
            log_loss += _loss
            print(train_info, end='', flush=True)
            if self.args.lr_scheduler == 'warmup':
                self.lr_scheduler['g'].step()
                self.lr_scheduler['d_a'].step()
                self.lr_scheduler['d_b'].step()

        if self.args.lr_scheduler == 'StepLR':
            self.lr_scheduler['g'].step()
            self.lr_scheduler['d_a'].step()
            self.lr_scheduler['d_b'].step()

        log_loss /= len(self.train_loader)
        self.model.eval()
        with torch.no_grad():
            self.scorer.score_log()
            for (_index, (plct, cbct, volume_id)) in enumerate(self.test_loader):
                plct = plct.reshape([-1, 1, 256, 256])
                cbct = cbct.reshape([-1, 1, 256, 256])

                lpuq = self.model.netG_A2B(cbct)
                self.scorer.cal(lpuq.detach().cpu(), plct.detach().cpu())
            log = self.scorer.mean_score(len(self.test_loader))

            log['ep'] = ep
            log['d_loss'] = log_loss[0].item()
            log['g_loss'] = log_loss[1].item()
            log['d_loss_a'] = log_loss[2].item()
            log['d_loss_b'] = log_loss[3].item()
            log['loss_idt'] = log_loss[4].item()
            log['loss_gan'] = log_loss[5].item()
            log['loss_cycle'] = log_loss[6].item()
            print(log)
            return log

    def eval(self, save_output=False):
        self.model.eval()
        with torch.no_grad():
            self.scorer.score_log()
            for (_index, (plct, cbct, volume_id)) in enumerate(self.test_loader):
                plct = plct.reshape([-1, 1, 256, 256])
                cbct = cbct.reshape([-1, 1, 256, 256])

                lpuq = self.model.netG_A2B(cbct)
                self.scorer.cal(lpuq.detach().cpu(), plct.detach().cpu())
                if save_output:
                    self.save_nii(lpuq, os.path.join(self.log_path, 'nii', f'{volume_id.item()}.nii'))
            log = self.scorer.mean_score(len(self.test_loader))
            print(log)
