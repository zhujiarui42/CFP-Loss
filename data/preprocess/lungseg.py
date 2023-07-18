import numpy as np
import SimpleITK as sitk
from data.preprocess.preprocess_functions import crop_3D_2
from data.preprocess import mask
import os
import glob
import cv2


save_path = "data/preprocesseddata"

data_path = glob.glob("data")
data_path_ct = glob.glob("data/*ct.nii")

k = 0
for cts in  data_path_ct:
    k +=1
    #read image
    id = cts[-7:-4] + ".nii"
    cb = os.path.join(data_path, f"{id}cb.nii")
    cb_file = sitk.ReadImage(cb)
    ct_file = sitk.ReadImage(cts)
    ct_array = sitk.GetArrayFromImage(ct_file)
    spacing_cb = cb_file.GetSpacing()
    spacing_ct = ct_file.GetSpacing()
    direction_cb = cb_file.GetDirection()
    direction_ct = ct_file.GetDirection()
    origin_cb = cb_file.GetOrigin()
    origin_ct = ct_file.GetOrigin()
    cb_array = sitk.GetArrayFromImage(cb_file)
    ct_array = sitk.GetArrayFromImage(ct_file)
    # get mask
    segmentation = mask.apply(ct_array)
    segmentation[segmentation == 2] = 1
    #apply mask
    ct_array = ct_array * segmentation
    cb_array = cb_array * segmentation

    #crop
    cb_array_crop,ct_array_crop = crop_3D_2(cb_array,ct_array)

    # to Image data
    # ct_img = sitk.GetImageFromArray(ct_array)
    # cb_img = sitk.GetImageFromArray(cb_array)

    ct_img_crop = sitk.GetImageFromArray(ct_array_crop)
    cb_img_crop = sitk.GetImageFromArray(cb_array_crop)

    # set spacing
    # ct_img.SetSpacing(spacing_ct)
    # cb_img.SetSpacing(spacing_cb)
    #
    # ct_img.SetDirection(direction_ct)
    # cb_img.SetDirection(direction_cb)
    #
    # ct_img.SetOrigin(origin_ct)
    # cb_img.SetOrigin(origin_cb)

    ct_img_crop.SetSpacing(spacing_ct)
    cb_img_crop.SetSpacing(spacing_cb)
    ct_img_crop.SetDirection(direction_ct)
    cb_img_crop.SetDirection(direction_cb)
    ct_img_crop.SetOrigin(origin_ct)
    cb_img_crop.SetOrigin(origin_cb)

    sitk.WriteImage(ct_img_crop, os.path.join(save_path, f"{id}ct.nii"))
    sitk.WriteImage(cb_img_crop, os.path.join(save_path, f"{id}cb.nii"))