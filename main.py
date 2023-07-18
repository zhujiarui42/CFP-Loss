"""
main function
"""
import argparse
import random

import numpy as np
import os
import torch
from utils import *
from models import *
from trainer import *
import torch.nn as nn

def make_args():
    parse = argparse.ArgumentParser()
    # dataloader
    parse.add_argument("--batch_size", type=int, default=16)
    parse.add_argument("--data_path", type=str, default="data")
    # parse.add_argument("--data_split", type=str, default="data/info.csv")
    # model
    parse.add_argument("--model", type=str, default='carunet')
    # parse.add_argument("--depth", type=int, default=110)
    # train
    parse.add_argument("--seed", type=int, default=-1)
    parse.add_argument("--cuda", type=str, default='2')
    parse.add_argument("--lr", type=float, default=1.e-4)
    parse.add_argument("--loss_fn", type=str, default='mix')
    parse.add_argument("--weight_decay", type=float, default=1e-4)
    parse.add_argument("--monitor_metric", type=str, default='base_psnr')
    parse.add_argument("--monitor_metriccs", type=str, default='base_acc')
    parse.add_argument("--epochs", type=int, default=100)
    parse.add_argument("--step_size", type=int, default=10)
    parse.add_argument("--gamma", type=float, default=0.1)
    parse.add_argument("--lr_scheduler", type=str, default='warmup')
    parse.add_argument("--optim", type=str, default='Adam')
    parse.add_argument("--trainer", type=str, default='base')
    # output
    parse.add_argument("--version", type=str, default='base')
    return parse.parse_args()

if __name__ == "__main__":
    # =============================
    # setup and fix random seeds
    # =============================
    args = make_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    if args.seed == -1:
        args.seed = random.randint(1, 10000)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(args)
    # =============================
    # dataloader
    # =============================
    train_loader, test_loader = build_dataloader(args)
    # =============================
    # crate model
    # =============================
    if args.model == 'unet':
        model = UNet(args)
    elif args.model == 'resunet':
        model = ResUNet(args)
    elif args.model == 'carunet':
        model = CARUNet(args)
    elif args.model == 'gan':
        model = GAN()
    elif args.model == 'cyclegan':
        model = CycleGAN()
    elif args.model == 'prnet':
        model = PRNet()
    elif args.model == 'selfnet':
        model = SelfNet()
    elif args.model == 'mtfsnet':
        model = MTFSNet()
    else:
        raise ValueError('wrong model!')
    model = model.cuda()
    # model.load_state_dict(torch.load(f"output/{args.version}/model.npy")['state_dict'])
    # print(model)

    # =============================
    # crate optim and lr scheduler
    # =============================
    optim = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optim, len(train_loader))
    # =============================
    # crate scorer and loss
    # =============================
    scorer = Scorer()
    if args.loss_fn == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_fn == 'l1':
        criterion = nn.L1Loss()
    elif args.loss_fn == 'ssim':
        criterion = SSIMLoss()
    elif args.loss_fn == 'mix':
        criterion = MixLoss()
    elif args.loss_fn == 'cycleganloss':
        criterion = CycleGANLoss()
    elif args.loss_fn == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_fn == 'reg':
        criterion = RegLoss()
    elif args.loss_fn == 'gradnorm':
        criterion = GradNormLoss()
    elif args.loss_fn == 'CFP':
        criterion = CFPLoss().cuda()
    # crate trainer
    # =============================
    if args.trainer == 'gan':
        trainer = CycleGANTrainer(args=args, train_loader=train_loader, test_loader=test_loader,
                                  model=model, optim=optim, criterion=criterion, scorer=scorer,
                                  lr_scheduler=lr_scheduler)
    if args.trainer == 'cyclegan':
        trainer = CycleGANTrainer(args=args, train_loader=train_loader, test_loader=test_loader,
                                  model=model, optim=optim, criterion=criterion, scorer=scorer,
                                  lr_scheduler=lr_scheduler)
    elif args.trainer == 'prnet':
        trainer = RegTrainer(args=args, train_loader=train_loader, test_loader=test_loader,
                             model=model, optim=optim, criterion=criterion, scorer=scorer,
                             lr_scheduler=lr_scheduler)
    elif args.trainer == 'self':
        trainer = SelfTrainer(args=args, train_loader=train_loader, test_loader=test_loader,
                            model=model, optim=optim, criterion=criterion, scorer=scorer,
                            lr_scheduler=lr_scheduler)
    elif args.trainer == 'mtfsnet':
        trainer = MTFSTrainer(args=args, train_loader=train_loader, test_loader=test_loader,
                            model=model, optim=optim, criterion=criterion, scorer=scorer,
                            lr_scheduler=lr_scheduler)
    else:
        # only end to end generator such as Unet, ResUnet and CARUnet (base)
        trainer = Trainer(args=args, train_loader=train_loader, test_loader=test_loader,
                          model=model, optim=optim, criterion=criterion, scorer=scorer,
                          lr_scheduler=lr_scheduler)

    trainer.train()
    # trainer.eval(True)