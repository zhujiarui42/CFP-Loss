import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
from glob import glob

class Dataset(object):
    def __init__(self, args, train=True):
        self.args = args
        self.data_path = args.data_path
        self.train = train
        self.data_list = glob(os.path.join(self.data_path, "train/slice" if self.train else "test/volume", "*.npy"))

    def __getitem__(self, item):
        if self.train:
            return self.get_train_data(item)
        else:
            return self.get_test_data(item)

    def get_test_data(self, item):
        # read file
        cbct, plct = np.load(self.data_list[item])  # volume2
        # [l, h, w]
        cbct = torch.from_numpy(cbct).float()
        plct = torch.from_numpy(plct).float()
        cbct = cbct.unsqueeze(1).cuda()
        plct = plct.unsqueeze(1).cuda()
        return plct, cbct, None


    def get_train_data(self, item):
        # read file
        _item = item
        # cb, pl = np.load(os.path.join(self.data_path, 'volume', f'{volume}.npy'))  # volume2
        cb, pl = np.load(self.data_list[item])  # slice2
        # [h, w]
        # img aug
        flip_code = np.random.randint(1)
        point = np.random.randint(0, 44, 2)
        corr = np.random.randint(-180, 180)
        cbct = im_aug(cb, 300,flip_code, point, corr, self.train)                # 256*256 inputs
        plct = im_aug(pl, 300,flip_code, point, corr, self.train)

        # [c, h, w]
        cbct = torch.from_numpy(cbct).float()
        plct = torch.from_numpy(plct).float()
        cbct = cbct.unsqueeze(0).cuda()
        plct = plct.unsqueeze(0).cuda()
        return plct, cbct

    def __len__(self):
        count = len(self.data_list)
        return count

def im_aug(_x, up_size=300, flip_code=0, point=(0, 0), corr=0, train=True):
    if train is False:
        return _x

    x = np.zeros([up_size, up_size])
    X = _x.shape[0]
    Y = _x.shape[1]
    l = up_size - X
    x[l//2:-l//2, l//2:-l//2] = _x
    M = cv2.getRotationMatrix2D((up_size // 2, up_size // 2), corr, 1.0)
    x = np.flip(x, axis=flip_code)
    x = cv2.warpAffine(x, M, (up_size, up_size))

    return x[point[0]:point[0] + Y, point[1]:point[1] + Y]


def build_dataloader(args):
    train_set = Dataset(args, True)
    test_set = Dataset(args, False)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=1)
    return train_dataloader, test_dataloader