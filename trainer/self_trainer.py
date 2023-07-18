import os
import torch
from trainer.BaseTrainer import BaseTrainer

class SelfTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(SelfTrainer, self).__init__(**kwargs)
        pass

    def train_one_epoch(self, ep):
        self.model.train()
        print("-"*10, f" {ep} ", "-"*10)
        log_loss = torch.zeros([1])
        iter_loss = []
        for (_index, (plct, cbct)) in enumerate(self.train_loader):

            self.optim.zero_grad()
            out_plct = self.model(plct)
            loss = self.criterion(out_plct,plct)

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

                out_plct = self.model(plct)

                self.scorer.cal(out_plct.detach().cpu(), plct.detach().cpu())

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

                out_plct = self.model(plct)
                out_cbct = self.model(cbct)
                self.scorer.cal(out_plct.detach().cpu(), plct.detach().cpu())
                print(self.scorer.cal(out_cbct.detach().cpu(), plct.detach().cpu()))
                print(self.scorer.cal(out_plct.detach().cpu(), plct.detach().cpu()))

                if save_output:
                    self.save_nii(out_cbct, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_outcb.nii'))
                    self.save_nii(out_plct, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_outpl.nii'))
                    self.save_nii(cbct, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_cbct.nii'))
                    self.save_nii(plct, os.path.join(self.log_path, 'nii', f'{volume_id.item()}_plct.nii'))
                    print(f"nii{volume_id.item()}saved")


            log = self.scorer.mean_score(len(self.test_loader))
            print(log)


