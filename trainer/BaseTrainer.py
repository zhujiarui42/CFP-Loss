
"""
trainer
"""
import torch
import os
import pandas as pd
import SimpleITK as sitk
from utils.visual import draw


def recover(img1, re=True):
    # [L, 1, H, W]
    img1 = img1.reshape([-1, 256, 256])
    if re:
        img2 = ((img1 - 0.192) * 1000) / 0.192
    return img2


class BaseTrainer(object):
    def __init__(self, args, train_loader, test_loader, model, optim, criterion, scorer, lr_scheduler=None):
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.scorer = scorer
        self.lr_scheduler = lr_scheduler

        self.best_score = 0
        self.best_scorecs = 0

        self.log_path = os.path.join("output", self.args.version)
        if os.path.exists(self.log_path) is False:
            os.mkdir(self.log_path)
            os.mkdir(os.path.join(self.log_path, 'nii'))

        self.best_recorder = pd.DataFrame()

    def train(self,isclass = False):
        for ep in range(self.args.epochs):
            log = self.train_one_epoch(ep)
            if isclass is True:
                self.check_bestcs(log)
            else:
                self.check_best(log)

    def eval(self):
        raise NotImplementedError

    def check_best(self, log):
        if log[self.args.monitor_metric] > self.best_score:
            self.best_score = log[self.args.monitor_metric]
            print("-"*10, f" new best {self.args.monitor_metric} {log[self.args.monitor_metric]} ", "-"*10)
            # self.save_log(log)
            self.save_model(log)

    def check_bestcs(self, log):
        if log[self.args.monitor_metric] > self.best_score:
            if log[self.args.monitor_metriccs] > self.best_scorecs:
                self.best_score = log[self.args.monitor_metric]
                self.best_scorecs = log[self.args.monitor_metriccs]
                print("-"*10, f" new best {self.args.monitor_metric} {log[self.args.monitor_metric]} ;new best {self.args.monitor_metriccs} {log[self.args.monitor_metriccs]}", "-"*10)
                self.save_model(log)

    def save_model(self, log):
        save_path = os.path.join(self.log_path, "model.npy")
        torch.save({"state_dict": self.model.state_dict(),
                    "log": log}, save_path)

    def save_log(self, log):
        record_path = os.path.join(self.log_path, 'log.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(log, ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def save_iter_loss(self, iter_loss):
        """
        :param iter_loss: list
        :return: None
        """
        record_path = os.path.join(self.log_path, 'iter_loss.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(iter_loss, ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def train_one_epoch(self, ep):
        raise NotImplementedError

    @staticmethod
    def save_nii(img: torch.Tensor, path, re=True):
        # [L, 1, H, W]
        img = img.reshape([-1, 256, 256])
        img = img.detach().cpu().numpy()
        if re:
            img = ((img - 0.192) * 1000) / 0.192
        img = sitk.GetImageFromArray(img)
        sitk.WriteImage(img, path)

    @staticmethod
    def save_nii224(img: torch.Tensor, path, re=True):
        # [L, 1, H, W]
        img = img.reshape([-1, 224, 224])
        img = img.detach().cpu().numpy()
        if re:
            img = ((img - 0.192) * 1000) / 0.192
            # img = ((img/2 - 0.192) * 1000) / 0.192
        img = sitk.GetImageFromArray(img)
        sitk.WriteImage(img, path)

class Trainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        pass

    def train_one_epoch(self, ep):
        self.model.train()
        print("-"*10, f" {ep} ", "-"*10)
        log_loss = torch.zeros([6])
        iter_loss = []
        for (_index, (plct, cbct)) in enumerate(self.train_loader):

            self.optim.zero_grad()
            lpuq = self.model(cbct)
            loss = self.criterion(lpuq, plct)
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

                lpuq = self.model(cbct)

                self.scorer.cal(lpuq.detach().cpu(), plct.detach().cpu())

            log = self.scorer.mean_score(len(self.test_loader))

            log['ep'] = ep
            log['loss'] = log_loss[0].item()
            print(log)
            self.save_log(log)
            return log

    def eval(self, save_output=False):
        self.model.eval()

        with torch.no_grad():
            self.scorer.score_log()
            for (_index, (plct, cbct, volume_id)) in enumerate(self.test_loader):
                plct = plct.reshape([-1, 1, 256, 256])
                cbct = cbct.reshape([-1, 1, 256, 256])

                lpuq = self.model(cbct)

                if save_output:
                    self.save_nii(cbct, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_cbct.nii'))
                    self.save_nii(plct, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_plct.nii'))
                    self.save_nii(lpuq, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_sct.nii'))
                    print(f"{volume_id.item()}——saved")
                    pass

            log = self.scorer.mean_score(len(self.test_loader))
            print(log)





