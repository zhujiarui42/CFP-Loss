import os
import torch
from trainer.BaseTrainer import BaseTrainer
from models.module import SpatialTransformer
import numpy as np


class MTFSTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(MTFSTrainer, self).__init__(**kwargs)
        self.warp = SpatialTransformer((256, 256)).cuda()
        self.initial_task_loss = None

    def train_one_epoch(self, ep):
        self.model.train()
        print("-" * 10, f" {ep} ", "-" * 10)
        log_loss = torch.zeros([1])
        iter_loss = []
        for (_index, (plct, cbct)) in enumerate(self.train_loader):

            moving_cbct = self.moving(cbct)
            moving_plct = self.moving(plct)
            self.optim.zero_grad()

            cbct_feature = self.model.encoder_cbct.forward(cbct)[0]
            plct_feature = self.model.encoder_plct.forward(plct)[0]
            out_plct = self.model.decoder_plct.forward(plct_feature)
            out_cbct = self.model.decoder_cbct.forward(cbct_feature)
            prob = self.model.decoder_class.forward(cbct_feature,plct_feature)

            moving_cbct_feature = self.model.encoder_cbct.forward(moving_cbct)[0]
            moving_plct_feature = self.model.encoder_plct.forward(moving_plct)[0]

            dvf_plct = self.model.decoder_dvf.forward(torch.cat([cbct_feature, moving_plct_feature], dim=1))
            dvf_cbct = self.model.decoder_dvf.forward(torch.cat([plct_feature, moving_cbct_feature], dim=1))

            warp_plct = self.warp(moving_plct, dvf_plct)
            warp_cbct = self.warp(moving_cbct, dvf_cbct)

            y = {'recon_pl': plct, 'warp_cb': plct, 'warp_pl': cbct,'prob':prob}
            x = { 'recon_pl': out_plct,'warp_cb': warp_cbct, 'warp_pl': warp_plct, 'prob':prob}
            task_loss = self.criterion(x, y)
            weighted_task_loss = torch.mul(self.criterion.weights, task_loss)
            # ==========================================
            # Grad Norm
            # ==========================================
            if ep == 0:
                # set L(0)
                if torch.cuda.is_available():
                    self.initial_task_loss = task_loss.data.cpu()
                else:
                    self.initial_task_loss = task_loss.data
                self.initial_task_loss = self.initial_task_loss.numpy()

            loss = torch.sum(weighted_task_loss)
            loss.backward(retain_graph=True)

            self.criterion.weights.grad.data = self.criterion.weights.grad.data * 0.0

            W = self.model.get_last_shared_layer()

            norms = []
            for i in range(len(task_loss)):
                # get the gradient of this task loss with respect to the shared parameters
                gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                # compute the norm
                norms.append(torch.norm(torch.mul(self.criterion.weights[i], gygw[0])))
            norms = torch.stack(norms)

            # compute the inverse training rate r_i(t)
            # \curl{L}_i
            if torch.cuda.is_available():
                loss_ratio = task_loss.data.cpu().numpy() / self.initial_task_loss
            else:
                loss_ratio = task_loss.data.numpy() / self.initial_task_loss

            # r_i(t)
            inverse_train_rate = loss_ratio / np.mean(loss_ratio)
            # print('r_i(t): {}'.format(inverse_train_rate))

            # compute the mean norm \tilde{G}_w(t)
            if torch.cuda.is_available():
                mean_norm = np.mean(norms.data.cpu().numpy())
            else:
                mean_norm = np.mean(norms.data.numpy())
            # print('tilde G_w(t): {}'.format(mean_norm))

            # compute the GradNorm loss
            # this term has to remain constant
            alpha = 0.12  # hyperparameters
            constant_term = torch.tensor(mean_norm * (inverse_train_rate ** alpha), requires_grad=False)
            if torch.cuda.is_available():
                constant_term = constant_term.cuda()
            # print('Constant term: {}'.format(constant_term))
            # this is the GradNorm loss itself
            grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
            # print('GradNorm loss {}'.format(grad_norm_loss))

            # compute the gradient for the weights
            self.criterion.weights.grad = torch.autograd.grad(grad_norm_loss, self.criterion.weights)[0]

            self.optim.step()

            train_info = "\r{}\t{:.4f}\t".format(_index, loss)

            _loss = torch.tensor([loss]).detach()
            iter_loss.append(loss.detach().cpu().numpy())
            log_loss += _loss
            print(train_info, flush=True, end='')
            if self.args.lr_scheduler == 'warmup':
                self.lr_scheduler.step()

        self.save_iter_loss(iter_loss)
        if self.args.lr_scheduler == 'StepLR':
            self.lr_scheduler.step()
        log_loss /= len(self.train_loader)
        self.model.eval()
        #
        data_list = ['recon_pl', 'warp_cb', 'warp_pl', 'prob']
        with torch.no_grad():

            self.scorer.score_log(data_list)
            for (_index, (plct, cbct, volume_id)) in enumerate(self.test_loader):
                plct = plct.reshape([-1, 1, 256, 256])
                cbct = cbct.reshape([-1, 1, 256, 256])
                moving_cbct = self.moving(cbct)
                moving_plct = self.moving(plct)
                cbct_feature = self.model.encoder_cbct.forward(cbct)[0]
                plct_feature = self.model.encoder_plct.forward(plct)[0]
                moving_cbct_feature = self.model.encoder_cbct.forward(moving_cbct)[0]
                moving_plct_feature = self.model.encoder_plct.forward(moving_plct)[0]
                out_plct = self.model.decoder_plct.forward(plct_feature)
                dvf_plct = self.model.decoder_dvf.forward(torch.cat([plct_feature, moving_cbct_feature], dim=1))
                dvf_cbct = self.model.decoder_dvf.forward(torch.cat([cbct_feature, moving_plct_feature], dim=1))
                warp_plct = self.warp(moving_plct, dvf_plct)
                warp_cbct = self.warp(moving_cbct, dvf_cbct)
                prob = self.model.decoder_class(cbct_feature,plct_feature)
                x = {'recon_pl': out_plct, 'warp_cb': warp_cbct, 'warp_pl': warp_plct, 'prob':prob}
                y = {'recon_pl': plct, 'warp_cb' : plct, 'warp_pl' : cbct,'prob':prob}
                self.scorer.cal(x, y)

            log = self.scorer.mean_score(len(self.test_loader))

            log['ep'] = ep
            log['loss'] = log_loss[0].item()
            print(log)
            return log

    def eval(self, save_output=False):
        self.model.eval()
        data_list = ['recon_pl', 'warp_cb', 'warp_pl','prob']
        with torch.no_grad():
            self.scorer.score_log(data_list)
            for (_index, (plct, cbct, volume_id)) in enumerate(self.test_loader):
                plct = plct.reshape([-1, 1, 256, 256])
                cbct = cbct.reshape([-1, 1, 256, 256])
                moving_cbct = self.moving(cbct)
                moving_plct = self.moving(plct)
                cbct_feature = self.model.encoder_cbct.forward(cbct)[0]
                plct_feature = self.model.encoder_plct.forward(plct)[0]
                moving_cbct_feature = self.model.encoder_cbct.forward(moving_cbct)[0]
                moving_plct_feature = self.model.encoder_plct.forward(moving_plct)[0]
                out_plct = self.model.decoder_plct.forward(plct_feature)
                dvf_plct = self.model.decoder_dvf.forward(torch.cat([plct_feature, moving_cbct_feature], dim=1))
                dvf_cbct = self.model.decoder_dvf.forward(torch.cat([cbct_feature, moving_plct_feature], dim=1))
                warp_plct = self.warp(moving_plct, dvf_plct)
                warp_cbct = self.warp(moving_cbct, dvf_cbct)
                prob = self.model.decoder_class(cbct_feature, plct_feature)

                x = {'recon_pl': out_plct.cpu(), 'warp_cb': warp_cbct.cpu(), 'warp_pl': warp_plct.cpu(),'prob':prob.cpu()}
                y = {'recon_pl': plct.cpu(), 'warp_cb':plct.cpu(),'warp_pl':cbct.cpu(),'prob':prob.cpu()}

                self.scorer.cal(x, y)

                if save_output:
                    self.save_nii(cbct, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_cbct.nii'))
                    self.save_nii(plct, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_plct.nii'))
                    self.save_nii(warp_cbct, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_warp_cbct.nii'))
                    self.save_nii(warp_plct, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_warp_plct.nii'))
                    # self.save_nii(dvf_cbct, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_dvf_cbct.nii'), re=False)
                    # self.save_nii(dvf_plct, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_dvf_plct.nii'),re=False)
                    self.save_nii(out_plct, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_plct_recon.nii'))
                    pass

            # log = self.scorer.mean_score(len(self.test_loader))
            # print(log)

    def moving(self, x):
        B, C, H, W = x.size()
        _x = torch.zeros([266, 266]).to(x.device)
        for i in range(B):
            _x[5:-5, 5:-5] = x[i]
            point = torch.randint(0, 10, [2])
            x[i] = _x[point[0]:point[0] + 256, point[1]:point[1] + 256]

        return x