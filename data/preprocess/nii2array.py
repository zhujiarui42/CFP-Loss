import SimpleITK as sitk
from scipy.ndimage import zoom
import numpy as np
import os


def preprocess(raw_data_path,save_path):
    for i in range(101):
        i+=1
        volume_id = i
        cb_file = sitk.ReadImage(os.path.join(raw_data_path, f"{volume_id}_cb.nii.gz"))
        pl_file = sitk.ReadImage(os.path.join(raw_data_path, f"{volume_id}_ct.nii.gz"))  # x, y, z
        cb_value = sitk.GetArrayFromImage(cb_file)
        pl_value = sitk.GetArrayFromImage(pl_file)  # z, y, x
        shape_ref = cb_value.shape
        cb_value = zoom(cb_value, (1, 256 / shape_ref[1], 256 / shape_ref[2]), order=0)  # z, y, x
        pl_value = zoom(pl_value, (1, 256 / shape_ref[1], 256 / shape_ref[2]), order=0)
        cb_value[cb_value > 900] = 1000
        cb_value[cb_value < -800] = -1000
        pl_value[pl_value > 900] = 1000
        pl_value[pl_value < -800] = -1000
        cb_value = ((cb_value * 0.192) / 1000 + 0.192)
        pl_value = ((pl_value * 0.192) / 1000 + 0.192)

        np.save(os.path.join(save_path, f'volume/{i}.npy'), [cb_value, pl_value])
        vol = [cb_value, pl_value]
        print(cb_value.shape)
        print("-" * 10, f' volume:{i}.npy is saved ', '-' * 10,len(vol))
        n_slice = cb_value.shape[0]

        _data_path = os.path.join(save_path, 'slice', f"{i}")
        if os.path.exists(_data_path) is False:
            os.mkdir(_data_path)
        for j in range(n_slice):
            np.save(os.path.join(_data_path, f'{j}.npy'), [cb_value[j], pl_value[j]])
        print("-" * 10, f' {n_slice} slices in volume id {i} is saved ', '-' * 10)


data_path = "data/preprocesseddata"
save_path = "data/npy"
preprocess(data_path,save_path)