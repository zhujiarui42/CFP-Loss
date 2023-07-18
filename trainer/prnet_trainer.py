import os
import torch
from trainer.BaseTrainer import BaseTrainer
from models.WarpST import SpatialTransformer
from utils.visual import draw


class Trainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        self.warp = SpatialTransformer((256, 256)).cuda()

    def train_one_epoch(self, ep):
        self.model.train()
        print("-"*10, f" {ep} ", "-"*10)
        log_loss = torch.zeros([1])
        iter_loss = []
        for (_index, (plct, cbct)) in enumerate(self.train_loader):

            cbct = self.moving(cbct)
            self.optim.zero_grad()
            filed = self.model(cbct, plct)
            warp_cbct = self.warp(cbct, filed)

            loss = self.criterion(warp_cbct, plct, filed)
            loss.backward()
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
        with torch.no_grad():
            self.scorer.score_log()
            for (_index, (plct, cbct, volume_id)) in enumerate(self.test_loader):
                plct = plct.reshape([-1, 1, 256, 256])
                cbct = cbct.reshape([-1, 1, 256, 256])

                filed = self.model(cbct, plct)
                warp_cbct = self.warp(cbct, filed)

                self.scorer.cal(warp_cbct.detach().cpu(), plct.detach().cpu())

            log = self.scorer.mean_score(len(self.test_loader))

            log['ep'] = ep
            log['loss'] = log_loss[0].item()
            print(log)
            return log

    def eval(self, save_output=False):
        self.model.eval()

        with torch.no_grad():
            self.scorer.score_log()
            for (_index, (plct, cbct, volume_id)) in enumerate(self.test_loader):
                plct = plct.reshape([-1, 1, 256, 256])
                cbct = cbct.reshape([-1, 1, 256, 256])

                filed = self.model(cbct, plct)
                warp_cbct = self.warp(cbct, filed)

                draw(warp_cbct[128, 0])
                draw(plct[128, 0])
                draw(cbct[128, 0])

                self.scorer.cal(warp_cbct.detach().cpu(), plct.detach().cpu())
                if save_output:
                    self.save_nii(warp_cbct, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_warp_cbct.nii'))
                    self.save_nii(filed, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_filed.nii'), re=False)
                    # # self.save_nii(cbct, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_cbct.nii'))

            log = self.scorer.mean_score(len(self.test_loader))
            print(log)

    def moving(self, x):
        B, C, H, W = x.size()
        _x = torch.zeros([266, 266]).to(x.device)
        for i in range(B):
            _x[5:-5, 5:-5] = x[i]
            point = torch.randint(0, 10, [2])
            x[i] = _x[point[0]:point[0] + 256, point[1]:point[1] + 256]

        return x
