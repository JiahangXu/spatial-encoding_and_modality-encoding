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

def train_net(net, device, epochs=5, batch_size=64, lr=0.001, nifti_train=-1,
              data_dict="labeled_bSSFP_Dict_byIndex.npy", dir_mark='default',
              save_cp=True, load_layer=[], model_mark='test', dcwt=0.8, loaddecay=0.25,
              rvwt=0.45):
    torch.manual_seed(3399)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3399)

    ## ----generate the DataLoader---- ##
    train = LabeledDataLoader2D("../Dataset/" + data_dict, nifti_train=nifti_train)
    val = LabeledDataLoader2D("../Dataset/" + data_dict, mode='val')
    test = LabeledDataLoader2D("../Dataset/" + data_dict, mode='test')

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val, batch_size=1, shuffle=False,
                            num_workers=1, pin_memory=True)
    test_loader = DataLoader(test, batch_size=1, shuffle=False,
                            num_workers=8, pin_memory=True)

    writer = SummaryWriter('runs/rs_%s/%s' % (dir_mark, model_mark))
    global_step = 0
    val_score = []

    load_params = []
    for layer in load_layer:
        load_params += eval('net.' + layer).parameters()
    load_params = list(map(id, load_params))

    optimizer = torch.optim.RMSprop([{'params': filter(lambda p: id(p) in load_params and p.requires_grad, net.parameters()),
                                   'lr': loaddecay * lr},
                                  {'params': filter(lambda p: id(p) not in load_params and p.requires_grad, net.parameters()),
                                   'lr': lr}])
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)
    dicewt = [0, 0.9-rvwt, 0.1, rvwt]

    logging.info(f'''Starting training segmentation network:
            Epochs:                 {epochs}
            Batch size:             {batch_size}
            Learning rate:          {lr} with exp decay 0.9999
            Lr for load layers:     {loaddecay * lr} with exp decay 0.9999
            Dice weight:            {dicewt}
            Train/Val/Test size:    {len(train)} / {len(val)} / {len(test)}
            Train/Val/Test image:   {train.nifti_train} / {val.nifti_train} / {test.nifti_train}
            Dataset:                {data_dict}
            weight of Dice:         {dcwt}
            Device:                 {device.type}
        ''')

    criterion1 = nn.MSELoss()
    criterion2 = MulticlassDiceLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            imgs = batch['image']
            true_masks = batch['mask']
            assert imgs.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            masks_pred = net(imgs)
            loss1 = criterion1(masks_pred.view(-1), true_masks.view(-1))
            loss2 = criterion2(masks_pred, true_masks, weights=dicewt)
            loss = (1 - dcwt) * loss1 + dcwt * loss2
            # logging.info('[%03d/%03d][%03d/%03d] Segmentation MSE / Dice: %.4f / %.4f;' % (
            #              epoch, epochs, i, len(train_loader), loss1.item(), loss2.item()))
            epoch_loss += loss.item()
            writer.add_scalar('Train/MseLoss', loss1.item(), global_step)
            writer.add_scalar('Train/DiceLoss', loss2.item(), global_step)
            writer.add_scalar('Train/TotalLoss', loss.item(), global_step)

            loss.backward()
            lr_schedule.step()
            optimizer.step()

            global_step += 1



        if True:
            # validation after the epoch finished
            [dice_200, dice_500, dice_600], [HD_LV_endo, HD_LV_epi, HD_RV_endo] = \
                eval_net(net, val_loader, device, epoch, dir_mark, model_mark)
            writer.add_scalar('Validation/Dice_200', dice_200, epoch)
            writer.add_scalar('Validation/Dice_500', dice_500, epoch)
            writer.add_scalar('Validation/Dice_600', dice_600, epoch)

            val_score.append([epoch, np.dot([dice_200, dice_500, dice_600], [0.45, 0.1, 0.45])])

            if epoch % 80 == 0:
                writer.add_scalar('Validation/HD_LV_endo', HD_LV_endo, epoch)
                writer.add_scalar('Validation/HD_LV_epi', HD_LV_epi, epoch)
                writer.add_scalar('Validation/HD_RV_endo', HD_RV_endo, epoch)

        if save_cp:
            torch.save(net.state_dict(),
                       'results/rs_%s/model_%s/checkpoints/' % (dir_mark, model_mark) + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()
    # print(val_score)
    sorted_val_score = sorted(val_score, key=lambda x: x[1], reverse=True)[:10]
    # print(sorted_val_score)
    sorted_val_score = sorted(sorted_val_score, key=lambda x: x[0])
    logging.info(f'Training Finished, the best validation score is %.4f in epoch %d!' % (
        sorted_val_score[0][1], sorted_val_score[0][0]))

    # test
    epoch_list = []
    ave_best_score = 0
    for best_epoch, best_score in sorted_val_score:
        best_model = torch.load('results/rs_%s/model_%s/checkpoints/' % (dir_mark, model_mark) + f'CP_epoch{best_epoch + 1}.pth',
                              map_location=device)
        net.load_state_dict(best_model)
        [dice_200, dice_500, dice_600], [HD_LV_endo, HD_LV_epi, HD_RV_endo] = \
            eval_net(net, test_loader, device, best_epoch, dir_mark, model_mark, mode='test')

        epoch_list.append(best_epoch)
        ave_best_score += best_score
        writer.add_scalar('Test/Dice_200', dice_200, best_epoch)
        writer.add_scalar('Test/Dice_500', dice_500, best_epoch)
        writer.add_scalar('Test/Dice_600', dice_600, best_epoch)

        writer.add_scalar('Test/HD_LV_endo', HD_LV_endo, best_epoch)
        writer.add_scalar('Test/HD_LV_epi', HD_LV_epi, best_epoch)
        writer.add_scalar('Test/HD_RV_endo', HD_RV_endo, best_epoch)

    val2txt = dir_mark + ', ' + model_mark + ', ' + str(round(ave_best_score / 10, 5)) + ', ' + str(epoch_list)
    with open('validation_score.txt', 'a') as file_handle:
        file_handle.write(val2txt)  # 写入
        file_handle.write('\n')

    # 移除其他pth文件，节约存储空间
    # print("save epoch .pth file: ", epoch_list)
    for temp_epoch in range(epochs):
        if temp_epoch not in epoch_list and os.path.exists(
                'results/rs_%s/model_%s/checkpoints/' % (dir_mark, model_mark) + f'CP_epoch{temp_epoch + 1}.pth'):
            os.remove('results/rs_%s/model_%s/checkpoints/' % (dir_mark, model_mark) + f'CP_epoch{temp_epoch + 1}.pth')


def eval_net(net, loader, device, epoch, dir_mark, model_mark, mode='val'):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    val_dice_200, val_dice_500, val_dice_600 = [], [], []
    HD_LV_endo_list, HD_LV_epi_list, HD_RV_endo_list = [], [], []
    dice = MulticlassDiceLoss()
    mode_mark = 'Validation' if mode == 'val' else 'Testing'

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


            if mode == 'val' and epoch % 80 == 0:
                # Husdorff Distance
                HD_LV_endo, HD_LV_epi, HD_RV_endo = multi_hausdorff_distance(mask_preds, true_masks)
                if HD_LV_endo != -1:  HD_LV_endo_list.append(HD_LV_endo)
                if HD_LV_epi != -1:  HD_LV_epi_list.append(HD_LV_epi)
                if HD_RV_endo != -1:  HD_RV_endo_list.append(HD_RV_endo)

            if mode == 'test':  # in test mode: save pred mask image
                # Husdorff Distance
                HD_LV_endo, HD_LV_epi, HD_RV_endo = multi_hausdorff_distance(mask_preds, true_masks)
                HD_LV_endo_list.append(HD_LV_endo)
                HD_LV_epi_list.append(HD_LV_epi)
                HD_RV_endo_list.append(HD_RV_endo)

                # if i % 10 == 0:
                #     img_Title = 'Dice for 200/500/600: %.4f / %.4f / %.4f, HD: %.4f / %.4f / %.4f' % (
                #         dice_200, dice_500, dice_600, HD_LV_endo, HD_LV_epi, HD_RV_endo)
                #     save_image_label(imgs, true_masks, mask_preds,
                #                      'results/rs_%s/model_%s/test_results/result_epoch%03d_fig%03d.png' % (
                #                          dir_mark, model_mark, epoch, i // 4), [0, 100, 200, 255], img_Title)

        val_dice_ave = [np.mean(i) for i in [val_dice_200, val_dice_500, val_dice_600]]
        val_HD_ave = []
        if mode == 'val' and epoch % 80 == 0:
            for list in [HD_LV_endo_list, HD_LV_epi_list, HD_RV_endo_list]:
                list_new = [i for i in list if i != -1]
                if len(list_new) != 0:
                    val_HD_ave.append(np.mean(list_new))
                else:
                    val_HD_ave.append(100)
            # logging.info('%s Dice for 200/500/600: %.4f / %.4f / %.4f, HD: %.4f / %.4f / %.4f' % (
            #     mode_mark, val_dice_ave[0], val_dice_ave[1], val_dice_ave[2],
            #     val_HD_ave[0], val_HD_ave[1], val_HD_ave[2]))

        if mode == 'test':
            for list in [HD_LV_endo_list, HD_LV_epi_list, HD_RV_endo_list]:
                list_new = [i for i in list if i != -1]
                if len(list_new) != 0:
                    val_HD_ave.append(np.mean(list_new))
                else:
                    val_HD_ave.append(100)
            result = pd.DataFrame()
            result['dice_200'], result['dice_500'], result['dice_600'] = \
                val_dice_200, val_dice_500, val_dice_600
            result['HD_LV_endo'], result['HD_LV_epi'], result['HD_RV_endo'] = \
                HD_LV_endo_list, HD_LV_epi_list, HD_RV_endo_list
            result.to_csv('results/rs_%s/model_%s/test_result_epoch%3d.csv' % (dir_mark, model_mark, epoch))
            logging.info('%s Dice for 200/500/600: %.4f / %.4f / %.4f, HD: %.4f / %.4f / %.4f in epoch %d' % (
                mode_mark, val_dice_ave[0], val_dice_ave[1], val_dice_ave[2],
                val_HD_ave[0], val_HD_ave[1], val_HD_ave[2], epoch))


    return (val_dice_ave, val_HD_ave if len(val_HD_ave) != 0 else [0,0,0])



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=500,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-n', '--traing-size', metavar='N', type=int, nargs='?', default=-1,
                        help='length of training set', dest='ntrain')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-l', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-m', '--model-mark', dest='modelmark', type=str, default='test',
                        help='string suffix to mark the model')
    parser.add_argument('-dm', '--dictionary-mark', dest='dirmark', type=str, default='default',
                        help='string suffix to mark the result dictionary')
    parser.add_argument('-dd', '--data-dict', dest='datadict', type=str,
                        default='labeled_bSSFP_Dict_byIndex.npy', help='dataset file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-f', '--frozen', dest='frozen', action='store_true', default=False,
                        help='Frozen parameters of some layers')
    parser.add_argument('-dwt', '--dice-weight', dest='diceweight', type=float, default=0.5,
                        help='weight of dice loss')
    parser.add_argument('-ld', '--load-decay', dest='loaddecay', type=float, default=0.25,
                        help='learning rate decay for loaded layer')
    parser.add_argument('-rvwt', '--rv-weight', dest='rvwt', type=float, default=0.45,
                        help='dice weight of the rv labels')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    modelmark = time.strftime('Mar%d_%H-%M-',time.localtime(time.time())) + \
                f'Unet_LR_{args.lr}_BS_{args.batchsize}_EPOCH_{args.epochs}_NTrain_{args.ntrain}_' + \
                args.modelmark

    logging.info(f'Model mark : {modelmark}')

    try:
        os.mkdir("results/rs_%s/" % args.dirmark)
        # print("results/rs_%s/" % args.dirmark)
    except OSError:
        pass

    try:
        os.mkdir("results/rs_%s/model_%s/" % (args.dirmark, modelmark))
        # print("results/rs_%s/model_%s/" % (args.dirmark, modelmark))
        os.mkdir('results/rs_%s/model_%s/checkpoints/' % (args.dirmark, modelmark))
        # print('results/rs_%s/model_%s/checkpoints/' % (args.dirmark, modelmark))
        os.mkdir('results/rs_%s/model_%s/test_results/' % (args.dirmark, modelmark))
        # print('results/rs_%s/model_%s/test_results/' % (args.dirmark, modelmark))
    except OSError:
        pass



    net = UNet(n_channels=1, n_classes=4)
    # logging.info(f'Network:\n'
    #              f'\t{net.n_channels} input channels\n'
    #              f'\t{net.n_classes} output channels (classes)\n'
    #              f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')
    # for k in net.state_dict():
    #     print(k, net.state_dict()[k].shape)
    #

    if args.load:
        netG_model = torch.load(args.load, map_location=device)
        netSeg_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in netG_model.items() if k in netSeg_dict}
        netSeg_dict.update(pretrained_dict)
        net.load_state_dict(netSeg_dict)
        logging.info(f'Model loaded from {args.load}')
        # for k, v in netG_model.items():
        #     print(k)
        # for k in netSeg_dict:
        #     print(k)
        load_layer = ["inc", "down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4"]

    else:
        load_layer = []

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True


    # if args.frozen:
    #     for param in net.parameters():
    #         param.requires_grad = False
    #     for layer in [net.up3, net.up4, net.outc]:
    #         for param in layer.parameters():
    #             param.requires_grad = True
    #     print("Frozen layers sucessfully: except net.up4, net.outc") # change here
    #
    #
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  nifti_train=args.ntrain,
                  load_layer=load_layer,
                  model_mark=modelmark,
                  dir_mark=args.dirmark,
                  dcwt=args.diceweight,
                  data_dict=args.datadict,
                  loaddecay=args.loaddecay,
                  rvwt=args.rvwt)
    except KeyboardInterrupt:
        logging.info('Keyboard Interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
