
"""
optimizers and lr scheduler
"""
import torch
import math


def build_optimizer(args, model):
    if args.model == 'cyclegan':
        return build_cycle_optimizer(args, model)
    elif args.model == 'dd_trans':
        return build_base_optimizer(args, model.trans)
    else:
        return build_base_optimizer(args, model)


def build_cycle_optimizer(args, model):
    optimizer_g = getattr(torch.optim, args.optim)([{"params": model.netG_A2B.parameters()},
                                    {"params": model.netG_B2A.parameters()}],
                                   lr=args.lr, betas=(0.5, 0.999))
    optimizer_d_a = getattr(torch.optim, args.optim)(model.netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d_b = getattr(torch.optim, args.optim)(model.netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))
    return {"g": optimizer_g, "d_a": optimizer_d_a, "d_b": optimizer_d_b}


def build_base_optimizer(args, model):
    optimizer = getattr(torch.optim, args.optim)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def build_lr_scheduler(args, optimizer, step_in_one_epoch):
    if args.lr_scheduler == 'warmup':
        lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                       args.step_size * step_in_one_epoch,
                                                       args.epochs * step_in_one_epoch)
    else:
        lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    print(f"Build {args.lr_scheduler} for {args.optim} in {args.version}")
    return lr_scheduler


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
   Warmup预热学习率：先从一个较小的学习率线性增加至原来设置的学习率，再进行学习率的线性衰减

    当 current_step < num_warmup_steps时，
    new_lr =current_step/num_warmup_steps * base_lr
    当current_step >= num_warmup_steps时，
    new_lr =(num_training_steps - current_step) / (num_training_steps -num_warmup_steps) * base_lr

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_line(current_step: int):
        # 自定义函数
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    def lr_lambda(current_step: int):
        # 自定义函数
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    def lr_cosine(current_step: int):
        # linear warmup
        end_lr = 0.001
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # cosine annealing decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_lr = max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))
        # lr = max(0.0, cosine_lr * (base_lr - end_lr) + end_lr)
        lr = max(0.0, cosine_lr * (1 - end_lr) + end_lr)
        return lr

    if type(optimizer) == dict:
        return {k: torch.optim.lr_scheduler.LambdaLR(v, lr_cosine, last_epoch) for k, v in optimizer.items()}
    else:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_cosine, last_epoch)



