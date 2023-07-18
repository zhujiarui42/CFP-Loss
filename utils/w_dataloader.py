
"""
dataloader for enhance and registration
enhance: select n slice continually
registration: random select n slice
"""
import torch
import pandas as pd
from torch.utils.data import DataLoader
import SimpleITK as sitk
from scipy.ndimage import zoom
import numpy as np
import os
import cv2


class Dataset(object):
    def __init__(self, args, train=True):
        self.args = args
        self.data_path = args.data_path
        self.train = train

        split = pd.read_csv(args.data_split)
        self.data_list = split[split['train'] == train]
        self.data_list.reset_index(drop=True, inplace=True)

    def __getitem__(self, item):
        if self.train:
            return self.get_train_data(item)
        else:
            return self.get_test_data(item)

    def get_test_data(self, item):
        # read file
        volume = self.data_list.loc[item, 'volume_id']
        cbct, plct = np.load(os.path.join(self.data_path, 'volume', f'{volume}.npy'))  # volume2
        # [l, h, w]
        cbct = torch.from_numpy(cbct).float()
        plct = torch.from_numpy(plct).float()
        cbct = cbct.unsqueeze(1).cuda()
        plct = plct.unsqueeze(1).cuda()
        return plct, cbct, volume


    def get_train_data(self, item):
        # read file
        _item = item
        i = 0
        while _item >= 0:
            if _item - self.data_list.loc[i, 'count'] < 0:
                break
            else:
                _item -= self.data_list.loc[i, 'count']
                i += 1
        volume = self.data_list.loc[i, 'volume_id']
        # cb, pl = np.load(os.path.join(self.data_path, 'volume', f'{volume}.npy'))  # volume2
        cb, pl = np.load(os.path.join(self.data_path, 'slice', f'{volume}', f'{_item}.npy'))  # slice2
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
        if self.train:
            count = self.data_list['count'].values.sum()
        else:
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

def preprocess(raw_data_path,save_path):
    # df = pd.read_csv("/home/servicing/projects/jiarui/CBCT_Jiarui/data/unzip_info_test.csv")
    for i in range(101):
        i+=1
        volume_id = i
        cb_file = sitk.ReadImage(os.path.join(raw_data_path, f"{volume_id}_cb.nii.gz"))
        pl_file = sitk.ReadImage(os.path.join(raw_data_path, f"{volume_id}_ct.nii.gz"))  # x, y, z
        cb_value = sitk.GetArrayFromImage(cb_file)
        pl_value = sitk.GetArrayFromImage(pl_file)  # z, y, x
        # spacing = pl_file.GetSpacing()
        shape_ref = cb_value.shape
        cb_value = zoom(cb_value, (1, 256 / shape_ref[1], 256 / shape_ref[2]), order=0)  # z, y, x
        pl_value = zoom(pl_value, (1, 256 / shape_ref[1], 256 / shape_ref[2]), order=0)
        cb_value[cb_value > 900] = 1000
        cb_value[cb_value < -800] = -1000
        pl_value[pl_value > 900] = 1000
        pl_value[pl_value < -800] = -1000
        cb_value = ((cb_value * 0.192) / 1000 + 0.192)
        pl_value = ((pl_value * 0.192) / 1000 + 0.192)

        np.save(os.path.join(save_path, f'volume/{i}.npy'), [cb_value, pl_value])  # volume2
        vol = [cb_value, pl_value]
        print(cb_value.shape)
        print("-" * 10, f' volume:{i}.npy is saved ', '-' * 10,len(vol))
        n_slice = cb_value.shape[0]
        # df['count'][df['volume_id']==i] = n_slice
        # assert n_slice == pl_value.shape[0]
        _data_path = os.path.join(save_path, 'slice', f"{i}")  ##slicer2
        if os.path.exists(_data_path) is False:
            os.mkdir(_data_path)
        for j in range(n_slice):
            np.save(os.path.join(_data_path, f'{j}.npy'), [cb_value[j], pl_value[j]])
        print("-" * 10, f' {n_slice} slices in volume id {i} is saved ', '-' * 10)
        # else :
        #     print("no mode specified")
        # df.to_csv("/home/servicing/projects/jiarui/CBCT_Jiarui/data/unzip_info_test.csv")     #csv test

# data_path = "/home/servicing/data/Cervical_CBCT"
# save_path = "/home/servicing/jiarui/cbct-all/data"
# preprocess(data_path,save_path)