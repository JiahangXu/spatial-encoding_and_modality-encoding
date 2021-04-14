import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time

from utils.utils import *
from model.model import SSNetD, SSNetG
# from model.model import SSNetG_layers as SSNetG

from tensorboardX import SummaryWriter
from utils.dataset import MetaUnlabeledDataLoader2D
from torch.utils.data import DataLoader, random_split

import warnings
warnings.filterwarnings('ignore')

def train_net(netG, netD, device, epochs=5, batch_size=128, lr=0.001, ntrain=-1,
              data_dict="all_Dict_byIndex.npy",
              val_percent=0.1, save_cp=True, model_mark='test', wtl2=0.998, task_style='all'):
    torch.manual_seed(3399)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3399)

    ## ----generate the DataLoader---- ##
    unlabeled_dataset = MetaUnlabeledDataLoader2D(task_list=['mmr', 'amb', 'amb2'])
    n_val = int(len(unlabeled_dataset) * val_percent)
    n_train = len(unlabeled_dataset) - n_val
    train, val = random_split(unlabeled_dataset, [n_train, n_val])
    if ntrain != -1:
        indices = torch.randperm(n_train)
        train = torch.utils.data.Subset(train, indices[:ntrain])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter('runs/%s' % model_mark)
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(train_loader) * batch_size}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    # Optimizers and Evaluation criterion
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-8)
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=0.05 * lr, betas=(0.5, 0.999), weight_decay=1e-8)

    criterionBCE = nn.BCELoss() # binary cross entropy, for discriminator

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    for epoch in range(epochs):
        netG.train()
        netD.train()

        for i, batch in enumerate(train_loader):

            img_tasks = batch['img_tasks']
            img_real = batch['img_real']
            task_mode = batch['task_mode']

            # Adversarial ground truths
            # real = Variable(Tensor(img_real.size(0), 1).fill_(1.0), requires_grad=False)
            # fake = Variable(Tensor(img_real.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            img_tasks = img_tasks.to(device=device, dtype=torch.float32)
            img_real = img_real.to(device=device, dtype=torch.float32)
            # task_mask = task_mask.to(device=device, dtype=torch.float32)

            # -----------------
            #  Train Generator
            # -----------------

            frozen_params(netD)
            optimizer_G.zero_grad()

            # Generate a batch of images
            # img_fake, _, _, _, _ = netG(img_tasks)
            img_fake = netG(img_tasks)

            # Loss measures generator's ability to fool the discriminator
            # lossG_D = criterionBCE(netD(img_fake), real)

            # lossG_l2 = criterionMSE(img_fake, img_real)
            # wtl2Matrix = torch.ones(img_real.shape).to(device=device, dtype=torch.float32) * 0.1
            # wtl2Matrix += task_mask * 0.9

            lossG_l2 = (img_fake - img_real).pow(2)
            # lossG_l2 = lossG_l2 * wtl2Matrix
            lossG = lossG_l2.mean()

            # writer.add_scalar('Generator/BCELoss', lossG_D.item(), global_step)
            writer.add_scalar('Generator/MSELoss', lossG.item(), global_step)

            lossG.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # free_params(netD)
            # optimizer_D.zero_grad()
            #
            # # Measure discriminator's ability to classify real from generated samples
            # lossD_real = criterionBCE(netD(img_real), real)
            # lossD_fake = criterionBCE(netD(img_fake), fake)
            logging.info('[%03d/%03d][%03d/%03d] Generator MSE: %.4f; ' % (
                epoch, epochs, i, len(train_loader), lossG.item()))
            # writer.add_scalar('Discriminator/BCELoss_real', lossD_real.item(), global_step)
            # writer.add_scalar('Discriminator/BCELoss_fake', lossD_fake.item(), global_step)
            #
            # lossD = (lossD_real + lossD_fake) / 2
            #
            # lossD.backward()
            # optimizer_D.step()

            global_step += 1

        # validation after the epoch finished
        val_l2 = eval_net(netG, netD, val_loader, device, n_val, epoch, model_mark)
        writer.add_scalar('Validation/MSELoss', val_l2, epoch)
        # writer.add_scalar('Validation/BCELoss_real', val_real, epoch)
        # writer.add_scalar('Validation/MSELoss_fake', val_fake, epoch)

        if save_cp:
            torch.save(netG.state_dict(),
                       'results/model_%s/checkpoints/' % model_mark + f'CP_netG_epoch{epoch + 1}.pth')
            torch.save(netD.state_dict(),
                       'results/model_%s/checkpoints/' % model_mark + f'CP_netD_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def eval_net(netG, netD, loader, device, n_val, epoch, model_mark):

    netG.eval()
    netD.eval()
    criterionBCE = nn.BCELoss()
    val_l2 = 0.
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    with torch.no_grad():
        for i, batch in enumerate(loader):

            img_tasks = batch['img_tasks']
            img_real = batch['img_real']
            # task_mask = batch['task_mask']
            task_mode = batch['task_mode']

            img_tasks = img_tasks.to(device=device, dtype=torch.float32)
            img_real = img_real.to(device=device, dtype=torch.float32)

            # generator loss for validation
            # img_fake, decoder1, decoder2, decoder3, decoder4 = netG(img_tasks)
            img_fake = netG(img_tasks)
            val_l2 += (img_fake - img_real).pow(2).mean()

            # # discriminator loss for validation
            # pred_label_real = netD(img_real)
            # pred_label_fake = netD(img_fake)
            #
            # val_real += criterionBCE(pred_label_real, Tensor(1, 1).fill_(1.0)).item()
            # val_fake += criterionBCE(pred_label_fake, Tensor(1, 1).fill_(0.0)).item()

            if i % 50 == 0 and epoch > 100:
                save_generate_image(img_real, img_tasks, img_tasks, img_fake,
                                    'results/model_%s/val_results/result_samples_epoch%03d_val%03d.png'
                                        % (model_mark, epoch, i),
                                    'Task: %s; L2 norm loss: %.4f; ' % (task_mode, val_l2))
                # for k, image in enumerate([decoder1, decoder2, decoder3, decoder4, img_fake]):
                #     save_decoder_image(image, 'results/model_%s/decoder/result_decoder%d_epoch%03d_val%03d'
                #                         % (model_mark, k + 1, epoch, i))

        logging.info('Validation L2 Norm Loss: %.4f;' % (val_l2 / n_val))

    return (val_l2 / n_val)


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=64,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-n', '--traing-size', metavar='N', type=int, nargs='?', default=-1,
                        help='length of training set', dest='ntrain')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-l', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=1.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-ts', '--task-style', dest='taskstyle', type=str, default='all',
                        help='claim task style: all, ccc, cpc, crc, sp')
    parser.add_argument('-m', '--model-mark', dest='modelmark', type=str, default='',
                        help='string suffix to mark the model')
    parser.add_argument('-w', '--weight-l2', metavar='W', type=float, default=1.0,
                        help='Weight of L2 loss in generator', dest='wtl2')
    parser.add_argument('-d', '--data-dict', dest='datadict', type=str,
                        default="all_Dict_byIndex.npy",
                        help='data dictionary for data loader')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    modelmark = time.strftime('Mar%d_%H-%M-',time.localtime(time.time())) + \
                f'SSnet_mt_TS_{args.taskstyle}_LR_{args.lr}_NTrain_{args.ntrain}_' + \
                args.modelmark
    logging.info(f'Model mark : {modelmark}')

    try:
        os.mkdir("results/model_%s" % modelmark)
        os.mkdir('results/model_%s/checkpoints/' % modelmark)
        os.mkdir('results/model_%s/val_results/' % modelmark)
        # os.mkdir('results/model_%s/decoder/' % modelmark)
    except OSError:
        pass

    # Initialize generator
    netG = SSNetG(n_channels=1, bilinear=True)
    if args.load:
        best_model = torch.load(args.load, map_location = device)
        netG.load_state_dict(best_model)
        logging.info(f'Model loaded from {args.load}')

    logging.info(f'Generator Network:\n'
                 f'\t{netG.n_channels} input channels\n'
                 f'\t{"Bilinear" if netG.bilinear else "Dilated conv"} upscaling')
    netG.to(device=device)


    # Initialize discriminator
    netD = SSNetD(n_channels=1)
    logging.info(f'Discriminator Network:\n'
                 f'\t{netD.n_channels} input channels\n')

    netD.to(device=device)


    try:
        train_net(netG=netG,
                  netD=netD,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  ntrain=args.ntrain,
                  device=device,
                  val_percent=args.val / 100,
                  model_mark=modelmark,
                  task_style=args.taskstyle,
                  wtl2=args.wtl2,
                  data_dict=args.datadict)
    except KeyboardInterrupt:
        logging.info('Keyboard interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
