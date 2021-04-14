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

from utils.loss import *
from utils.utils import *
from model.model import UNet

from tensorboardX import SummaryWriter
from utils.dataset import LabeledDataLoader2D
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

def val_net(net, data_dict, epoch_list, device, dir_mark, model_mark):
    torch.manual_seed(3399)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3399)

    ## ----generate the DataLoader---- ##
    val = LabeledDataLoader2D("../Dataset/" + data_dict, mode='val')

    val_loader = DataLoader(val, batch_size=1, shuffle=False,
                            num_workers=1, pin_memory=True)

    # test
    val_scores = []
    ave_best_score = 0
    for epoch in tqdm(epoch_list):
        val_model = torch.load(
            'results/%s/%s/checkpoints/' % (dir_mark, model_mark) + f'CP_epoch{epoch + 1}.pth',
            map_location=device)

        net.load_state_dict(val_model)
        val_score = eval_net(net, val_loader, device)
        val_scores.append([epoch, val_score])


    sorted_val_score = sorted(val_scores, key=lambda x: x[1], reverse=True)[:10]
    # print(sorted_val_score)
    sorted_val_score = sorted(sorted_val_score, key=lambda x: x[0])
    best_epoch, best_score = zip(*sorted_val_score)
    ave_best_score = np.mean(best_score)

    val2txt = dir_mark + ', ' + model_mark + ', ' + str(round(ave_best_score, 5)) + ', ' + str(best_epoch)
    print(val2txt)
    with open('validation_score.txt', 'a') as file_handle:
        file_handle.write(val2txt)  # 写入
        file_handle.write('\n')

    # 移除其他pth文件，节约存储空间
    # print("save epoch .pth file: ", epoch_list)
    for temp_epoch in range(500):
        if temp_epoch not in best_epoch and os.path.exists(
                'results/%s/%s/checkpoints/' % (dir_mark, model_mark) + f'CP_epoch{temp_epoch + 1}.pth'):
            print("rm %d" % temp_epoch)
            os.remove('results/%s/%s/checkpoints/' % (dir_mark, model_mark) + f'CP_epoch{temp_epoch + 1}.pth')


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    val_dice_200, val_dice_500, val_dice_600 = [], [], []
    dice = MulticlassDiceLoss()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            lab_preds = net(imgs)
            mask_preds = torch.zeros(lab_preds.shape).to(device)
            mask_preds = mask_preds.scatter_(1, lab_preds.argmax(1).unsqueeze(1), 1)
            # mask_preds = cca(mask_preds, true_masks, device=device)

            # dice
            dice_200 = 1 - dice(mask_preds, true_masks, weights=[0, 1, 0, 0]).item()
            dice_500 = 1 - dice(mask_preds, true_masks, weights=[0, 0, 1, 0]).item()
            dice_600 = 1 - dice(mask_preds, true_masks, weights=[0, 0, 0, 1]).item()
            val_dice_200.append(dice_200)
            val_dice_500.append(dice_500)
            val_dice_600.append(dice_600)

        val_dice_ave = [np.mean(i) for i in [val_dice_200, val_dice_500, val_dice_600]]
        val_score = np.dot(val_dice_ave, [0.45, 0.1, 0.45])

    return (val_score)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=1, n_classes=4)
    net.to(device=device)

    result_list = [["labeled_bSSFP_Dict_byIndex.npy", np.arange(500), "rs_no-pre_new",
                    "model_Feb01_00-50-Unet_LR_0.001_BS_36_EPOCH_500_NTrain_2_0.5Dice"],
                   ["labeled_bSSFP_Dict_byIndex.npy", np.arange(500), "rs_no-pre_new",
                    "model_Feb01_01-35-Unet_LR_0.001_BS_36_EPOCH_500_NTrain_2_0.6Dice"],
                   ["labeled_bSSFP_Dict_byIndex.npy", np.arange(500), "rs_no-pre_new",
                    "model_Feb01_02-21-Unet_LR_0.001_BS_36_EPOCH_500_NTrain_2_0.7Dice"],
                   ["labeled_bSSFP_Dict_byIndex.npy", np.arange(500), "rs_no-pre_new",
                    "model_Feb01_03-06-Unet_LR_0.001_BS_36_EPOCH_500_NTrain_2_0.8Dice"],
                   ["labeled_LGE_Dict_byIndex.npy", np.arange(500), "rs_no-pre_new",
                    "model_Feb01_03-53-Unet_LR_0.001_BS_36_EPOCH_500_NTrain_2_0.5Dice_LGE"],
                   ["labeled_LGE_Dict_byIndex.npy", np.arange(500), "rs_no-pre_new",
                    "model_Feb01_05-06-Unet_LR_0.001_BS_36_EPOCH_500_NTrain_2_0.6Dice_LGE"],
                   ["labeled_LGE_Dict_byIndex.npy", np.arange(500), "rs_no-pre_new",
                    "model_Mar01_16-58-Unet_LR_0.001_BS_36_EPOCH_500_NTrain_2_0.7Dice_LGE"]]

    for data_dict, epoch_list, dir_mark, model_mark in result_list:
        val_net(net, data_dict, epoch_list, device, dir_mark, model_mark)


