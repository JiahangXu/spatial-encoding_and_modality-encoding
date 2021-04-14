import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import collections
import pandas as pd
import nibabel as nib

from utils.loss import *
from utils.utils import *
from model.model import UNet
import random

from tensorboardX import SummaryWriter
from utils.dataset import LabeledDataLoader2D, testPatient42
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

def test_net(net, device, dir_mark1, dir_mark2, model_mark, modality,
             data_dict="labeled_bSSFP_Dict_byIndex.npy"):
    torch.manual_seed(3399)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3399)

    ## ----generate the DataLoader---- ##
    test = LabeledDataLoader2D("../Dataset/" + data_dict, mode='test')
    # test = testPatient42("../Dataset/" + data_dict, mode='test')
    test_loader = DataLoader(test, batch_size=1, shuffle=False,
                             num_workers=8, pin_memory=True)

    for filename in os.listdir(
            'results/%s/%s/%s/checkpoints/' % (dir_mark1, dir_mark2, model_mark)):
        if filename.endswith(".pth"):
            # print(dir_mark1, dir_mark2, model_mark, filename)

            slice_pred = {}
            best_model = torch.load('results/%s/%s/%s/checkpoints/' % (
                dir_mark1, dir_mark2, model_mark) + filename,
                map_location=device)
            net.load_state_dict(best_model)

            net.eval()

            with torch.no_grad():
                for batch in test_loader:
                    imgs = batch['image']
                    # for k, v in batch.items():
                    #     print(k)
                    slice_index = int(batch['index'])
                    # print(slice_index)
                    # true_masks = batch['mask']

                    imgs = imgs.to(device=device, dtype=torch.float32)

                    lab_preds = net(imgs)
                    mask_preds = torch.zeros(lab_preds.shape).to(device)
                    mask_preds = mask_preds.scatter_(1, lab_preds.argmax(1).unsqueeze(1), 1)
                    mask_preds = np.array(mask_preds.to('cpu'))
                    # if slice_index > 1123 and slice_index < 1169:
                    #     img_Title = 'slice_index'
                    #     save_image_label(imgs, true_masks, mask_preds,
                    #                      'result_epoch%03d.png' % (slice_index), [0, 100, 200, 255], img_Title)

                    mask_new = np.zeros((128,128))
                    for i, lab in zip(range(1,4), [200, 500, 600]):
                        mask_new[mask_preds[0, i] == 1] = lab

                    slice_pred[slice_index] = mask_new
            try:
                os.mkdir('NiFTI/%s/%s/%s/%s/' % (modality, dir_mark1, dir_mark2, filename))
            except OSError:
                pass
            out_dir = 'NiFTI/%s/%s/%s/%s/' % (modality, dir_mark1, dir_mark2, filename)
            out_nifti(data_dict, slice_pred, out_dir)
            # return(slice_pred)


def out_nifti(dictForSlice_dir, data, out_dir):

    origin_dict = pd.DataFrame(list(np.load("../Dataset/" + dictForSlice_dir, allow_pickle=True)))
    nifti_index_list = np.unique(origin_dict['nifti_index'])
    random.seed(3399)
    random.shuffle(nifti_index_list)
    nifti_index = nifti_index_list[int(len(nifti_index_list) - 30):]
    # nifti_index = [14, 16]

    for nifti_idx in nifti_index:
        temp = origin_dict[(origin_dict['nifti_index'] == nifti_idx) &
                           (origin_dict['rotation'] == 0)]
        # print(len(temp))
        img_info = temp.iloc[0]
        img_dir = img_info['img_dir']
        img_name = img_info['img_name']

        center_x = img_info['center_x']
        center_y = img_info['center_y']

        lab_dir = img_dir.replace('img', 'lab')
        lab_name = img_name.replace(".nii.gz", "_manual.nii.gz")
        lab = nib.load(os.path.join(lab_dir, lab_name))
        nii_data = lab.get_data()

        affine = lab.affine.copy()

        lab_pred = np.zeros(nii_data.shape)
        for i in range(len(temp)):
            temp_info = temp.iloc[i]
            lab_pred[int(center_x - 64): int(center_x + 64),
                     int(center_y - 64): int(center_y + 64),
                     temp_info['slice_index']] = data[temp_info['index']]

        lab_pred = lab_pred.astype('int16')
        new_nii = nib.Nifti1Image(lab_pred, affine)
        nib.save(new_nii, out_dir + img_name.replace(".nii.gz", "_pred.nii.gz"))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=4)
    net.to(device=device)

    try:
        os.mkdir("NiFTI")
    except OSError:
        pass

    waiting_list = []
    '''
    ############### NiFTI_86_1
    waiting_list = [["data_2", "rs_no-pre_new", "model_Feb01_03-53-Unet_LR_0.001_BS_36_EPOCH_500_NTrain_2_0.5Dice_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_2", "rs_no-pre_new", "model_Mar01_23-07-Unet_LR_0.0005_BS_36_EPOCH_500_NTrain_2_0.8Dice",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_2", "rs_sp_new", "model_Mar02_00-10-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_2_load_0.5Dice_sp",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_2", "rs_amb_new",
                     "model_Mar02_00-23-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_2_load_0.5Dice_map",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_2", "rs_amb_new",
                     "model_Mar02_02-47-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_2_load_0.6Dice_map_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_2", "rs_sp_new",
                     "model_Mar02_06-41-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_2_load_0.5Dice_sp_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_2", "rs_mmr_new",
                     "model_Mar02_12-48-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_2_0.5Dice_mmr_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_2", "rs_mmr_new", "model_Mar02_16-03-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_2_0.5Dice_mmr",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_2", "rs_ccc50", "model_Mar02_16-50-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_2_0.7Dice_ccc50",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_2", "rs_ccc50", "model_Mar02_18-48-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_2_0.7Dice_ccc50_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_2", "rs_amb1", "model_Mar03_00-25-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_2_0.6Dice_amb1_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_2", "rs_ccc64", "model_Mar03_02-36-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_2_0.7Dice_ccc64",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_2", "rs_amb1", "model_Mar03_03-52-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_2_0.7Dice_amb1",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_2", "rs_ccc64", "model_Mar03_04-02-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_2_0.6Dice_ccc64_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_2", "rs_cpc", "model_Mar03_16-30-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_2_0.6Dice_cpc",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_2", "rs_cpc", "model_Mar03_18-21-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_2_0.6Dice_cpc_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_10", "rs_amb_10",
                     "model_Mar06_20-51-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_10_load_0.6Dice_map",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_10", "rs_ccc50_10",
                     "model_Mar06_21-57-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_10_0.7Dice_ccc50",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_10", "rs_amb1_10",
                     "model_Mar07_00-03-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_10_0.5Dice_amb1_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_10", "rs_ccc50_10",
                     "model_Mar07_01-41-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_10_0.7Dice_ccc50_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_10", "rs_ccc64_10",
                     "model_Mar07_06-36-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_10_0.8Dice_ccc64",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_10", "rs_ccc64_10",
                     "model_Mar07_07-54-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_10_0.6Dice_ccc64_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_10", "rs_no-pre_10", "model_Mar07_14-48-Unet_LR_0.001_BS_36_EPOCH_500_NTrain_10_0.5Dice_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_10", "rs_no-pre_10", "model_Mar08_03-09-Unet_LR_0.0005_BS_36_EPOCH_500_NTrain_10_0.7Dice",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    
                    ["data_10", "rs_cpc_10", "model_Mar09_02-39-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_10_0.5Dice_cpc",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    
                    ["data_10", "rs_cpc_10",
                     "model_Mar09_05-03-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_10_0.5Dice_cpc_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_10", "rs_amb_10",
                     "model_Mar09_13-08-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_10_load_0.6Dice_map_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_10", "rs_amb1_10", "model_Mar09_14-10-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_10_0.6Dice_amb1",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_10", "rs_mmr_10",
                     "model_Mar10_01-21-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_10_0.8Dice_mmr_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_1", "rs_ccc64_1",
                     "model_Mar10_02-37-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_1_0.6Dice_ccc64_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_1", "rs_ccc64_1", "model_Mar10_03-55-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_1_0.5Dice_ccc64",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_1", "rs_amb_1",
                     "model_Mar10_06-50-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_1_load_0.6Dice_map_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_10", "rs_amb2_10",
                     "model_Mar10_08-04-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_10_load_0.7Dice_amb2_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_1", "rs_amb_1", "model_Mar10_09-59-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_1_load_0.5Dice_map",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_10", "rs_sp_10",
                     "model_Mar10_13-37-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_10_load_0.5Dice_sp_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_1", "rs_ccc50_1",
                     "model_Mar10_19-06-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_1_0.5Dice_ccc50_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_10", "rs_mmr_10", "model_Mar10_21-39-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_10_0.7Dice_mmr",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_1", "rs_ccc50_1", "model_Mar10_23-21-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_1_0.6Dice_ccc50",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_10", "rs_amb2_10",
                     "model_Mar11_02-39-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_10_load_0.6Dice_amb2",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_1", "rs_cpc_1", "model_Mar11_04-04-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_1_0.6Dice_cpc_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_1", "rs_cpc_1", "model_Mar11_06-37-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_1_0.5Dice_cpc",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_1", "rs_mmr_1", "model_Mar11_07-45-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_1_0.5Dice_mmr_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_10", "rs_sp_10", "model_Mar11_08-24-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_10_load_0.6Dice_sp",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_1", "rs_mmr_1", "model_Mar11_11-50-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_1_0.6Dice_mmr",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_1", "rs_sp_1",
                     "model_Mar11_16-40-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_1_load_0.7Dice_sp_LGE",
                     "labeled_LGE_Dict_byIndex.npy"],
                    ["data_1", "rs_sp_1", "model_Mar11_18-59-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_1_load_0.7Dice_sp",
                     "labeled_bSSFP_Dict_byIndex.npy"]]
    ############### NiFTI_1309_1
    waiting_list = [["data_1", "rs_no-pre_1", "model_Mar09_02-44-Unet_LR_0.0005_BS_72_EPOCH_500_NTrain_1_0.6Dice",
                     "labeled_bSSFP_Dict_byIndex.npy"],
                    ["data_1", "rs_no-pre_1", "model_Mar10_02-14-Unet_LR_0.0005_BS_36_EPOCH_500_NTrain_1_0.8Dice_LGE",
                     "labeled_LGE_Dict_byIndex.npy"]]
    ############### NiFTI_40_1
    waiting_list = [
        ["data_1", "rs_meta2_1", "model_Mar13_03-47-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_1_load_0.5Dice_meta2",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_1", "rs_meta2_1", "model_Mar12_17-43-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_1_load_0.6Dice_meta2_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_2", "rs_meta2_2", "model_Mar11_09-01-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_2_load_0.6Dice_meta2",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_2", "rs_meta2_2", "model_Mar10_21-28-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_2_load_0.7Dice_meta2_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_5", "rs_meta2_5", "model_Mar13_06-44-Unet_LR_0.001_BS_36_EPOCH_200_NTrain_5_load_0.5Dice_meta2",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_5", "rs_meta2_5", "model_Mar12_22-05-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_5_load_0.6Dice_meta2_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_10", "rs_meta2_10", "model_Mar11_12-00-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_10_load_0.6Dice_meta2",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_10", "rs_meta2_10", "model_Mar11_05-02-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_10_load_0.8Dice_meta2_LGE",
         "labeled_LGE_Dict_byIndex.npy"]]
    ############### NiFTI_86_4
    waiting_list = [
        ["data_2", "rs_amb2_2_backup",
         "model_Mar10_12-13-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_2_load_0.5Dice_amb2_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_2", "rs_meta3_2", "model_Mar11_13-00-Unet_LR_0.001_BS_72_EPOCH_200_NTrain_2_load_0.5Dice_meta2_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_10", "rs_meta3_10", "model_Mar11_19-34-Unet_LR_0.001_BS_72_EPOCH_200_NTrain_10_load_0.5Dice_meta2_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_5", "rs_sp_5", "model_Mar12_20-41-Unet_LR_0.001_BS_32_EPOCH_200_NTrain_5_load_0.7Dice_sp_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_5", "rs_mmr_5", "model_Mar12_15-06-Unet_LR_0.001_BS_32_EPOCH_200_NTrain_5_0.6Dice_mmr_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_5", "rs_ccc50_5", "model_Mar13_06-10-Unet_LR_0.0005_BS_32_EPOCH_200_NTrain_5_0.7Dice_ccc50_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_5", "rs_amb2_5", "model_Mar12_21-18-Unet_LR_0.0005_BS_32_EPOCH_200_NTrain_5_load_0.7Dice_amb2_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_5", "rs_cpc_5", "model_Mar12_07-26-Unet_LR_0.0005_BS_32_EPOCH_200_NTrain_5_0.6Dice_cpc_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_5", "rs_amb_5", "model_Mar12_03-06-Unet_LR_0.0005_BS_32_EPOCH_200_NTrain_5_load_0.5Dice_map_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_1", "rs_amb2_1", "model_Mar12_00-43-Unet_LR_0.0005_BS_64_EPOCH_200_NTrain_1_load_0.5Dice_amb2_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_1", "rs_meta3_1", "model_Mar13_02-20-Unet_LR_0.0005_BS_72_EPOCH_200_NTrain_1_load_0.8Dice_meta2_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_2", "rs_amb2_2_backup", "model_Mar10_16-54-Unet_LR_0.0005_BS_36_EPOCH_200_NTrain_2_load_0.6Dice_amb2",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_5", "rs_sp_5", "model_Mar13_16-47-Unet_LR_0.0005_BS_32_EPOCH_200_NTrain_5_load_0.5Dice_sp",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_5", "rs_mmr_5", "model_Mar13_16-24-Unet_LR_0.001_BS_32_EPOCH_200_NTrain_5_0.8Dice_mmr",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_5", "rs_ccc50_5", "model_Mar13_08-42-Unet_LR_0.001_BS_32_EPOCH_200_NTrain_5_0.8Dice_ccc50",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_5", "rs_amb2_5", "model_Mar12_23-09-Unet_LR_0.001_BS_32_EPOCH_200_NTrain_5_load_0.6Dice_amb2",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_5", "rs_cpc_5", "model_Mar13_12-38-Unet_LR_0.001_BS_32_EPOCH_200_NTrain_5_0.6Dice_cpc",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_5", "rs_amb_5", "model_Mar12_06-12-Unet_LR_0.001_BS_32_EPOCH_200_NTrain_5_load_0.6Dice_map",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_1", "rs_amb2_1", "model_Mar12_03-51-Unet_LR_0.0005_BS_32_EPOCH_200_NTrain_1_load_0.5Dice_amb2",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_1", "rs_meta3_1", "model_Mar12_06-58-Unet_LR_0.001_BS_72_EPOCH_200_NTrain_1_load_0.5Dice_meta3",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_2", "rs_meta3_2", "model_Mar12_09-58-Unet_LR_0.001_BS_72_EPOCH_200_NTrain_2_load_0.5Dice_meta3",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_10", "rs_meta3_10", "model_Mar13_10-52-Unet_LR_0.0005_BS_72_EPOCH_200_NTrain_10_load_0.5Dice_meta3",
         "labeled_bSSFP_Dict_byIndex.npy"]]
    '''
    waiting_list = [
        ["data_1", "rs_mmr_1_retry", "model_Mar13_18-54-Unet_LR_0.0005_BS_30_EPOCH_200_NTrain_1_0.5Dice_mmr_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_5", "rs_no-pre_5", "model_Mar13_22-40-Unet_LR_0.001_BS_32_EPOCH_500_NTrain_5_0.8Dice",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_5", "rs_meta3_5", "model_Mar14_04-42-Unet_LR_0.001_BS_32_EPOCH_200_NTrain_5_load_0.5Dice_meta3",
         "labeled_bSSFP_Dict_byIndex.npy"],
        ["data_5", "rs_no-pre_5", "model_Mar13_04-30-Unet_LR_0.0005_BS_32_EPOCH_500_NTrain_5_0.5Dice_LGE",
         "labeled_LGE_Dict_byIndex.npy"],
        ["data_5", "rs_meta3_5", "model_Mar12_02-07-Unet_LR_0.0005_BS_32_EPOCH_200_NTrain_5_load_0.6Dice_meta2_LGE",
         "labeled_LGE_Dict_byIndex.npy"]
    ]

    for dir_mark1, dir_mark2, model_mark, data_dict in tqdm(waiting_list):
        modality = "LGE" if data_dict == "labeled_LGE_Dict_byIndex.npy" else "bSSFP"
        if not os.path.exists('results/%s/%s/%s/checkpoints/' % (dir_mark1, dir_mark2, model_mark)):
            print('results/%s/%s/%s/checkpoints/' % (dir_mark1, dir_mark2, model_mark))

        try:
            os.mkdir("NiFTI/%s/" % modality)
        except OSError:
            pass

        try:
            os.mkdir("NiFTI/%s/%s/" % (modality, dir_mark1))
        except OSError:
            pass

        try:
            os.mkdir("NiFTI/%s/%s/%s/" % (modality, dir_mark1, dir_mark2))
        except OSError:
            pass

        test_net(net, device, dir_mark1, dir_mark2, model_mark, modality, data_dict=data_dict)


