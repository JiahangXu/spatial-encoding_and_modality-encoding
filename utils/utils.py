from __future__ import print_function, division, absolute_import, unicode_literals

from PIL import Image
from torch.utils.data.dataset import Dataset
import nibabel as nib
import glob
import math
from torch.autograd import Variable

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import os
import torch
import torch.nn as nn
import numpy as np
from torch import randperm
from torch._utils import _accumulate
from skimage.util import random_noise
import cv2


def save_image(batch_Imgs, save_path):
    # batch_Imgs: [nbatch, channel=1, nx, ny]
    batch_Imgs = np.array(batch_Imgs.to('cpu'))
    nbatch = batch_Imgs.shape[0]
    nrow = math.ceil(nbatch**0.5)
    fig, ax = plt.subplots(nrow, nrow, figsize=(12, 12),
                           sharey=True, sharex=True)
    ax = np.atleast_2d(ax)
    for i in range(nbatch):
        img = np.array(batch_Imgs[i].squeeze(0))
        img = (img + 1) / 2 * 255
        cax = ax[i//nrow, i-i//nrow*nrow].imshow(img.astype(np.uint8),cmap='gray')
    fig.tight_layout()
    fig.savefig(save_path)


def save_image_label(img, true_mask, mask_pred, save_path, intensity, title):
    """
    output image for validation (batch size=1)
    :param img:
    :param true_mask:
    :param mask_pred:
    :param save_path:
    :param intensity:
    :return:
    """
    # batch_Labs: [nbatch, channel=4, nx, ny]
    img, true_mask, mask_pred = np.array(img.squeeze(0).to('cpu')), \
                                   np.array(true_mask.squeeze(0).to('cpu')), \
                                   np.array(mask_pred.squeeze(0).to('cpu'))
    fig, ax = plt.subplots(1, 3, figsize=(24, 8),
                           sharey=True, sharex=True)
    ax = np.atleast_2d(ax)

    img = np.array(img.squeeze(0))
    img = (img + 1) / 2 * 255

    true_mask_new = np.zeros((true_mask.shape[1], true_mask.shape[2]))
    mask_pred_new = np.zeros((mask_pred.shape[1], mask_pred.shape[2]))
    for i in range(len(intensity)):

        true_mask_forClass = true_mask[i, :]
        true_mask_new[np.where(true_mask_forClass==1)] = intensity[i]

        mask_pred_forClass = mask_pred[i, :]
        mask_pred_new[np.where(mask_pred_forClass == 1)] = intensity[i]

    ax[0, 0].imshow(img.astype(np.uint8),cmap='gray')
    ax[0, 0].set_title('Image', fontsize=24)
    ax[0, 1].imshow(true_mask_new.astype(np.uint8), cmap='gray')
    ax[0, 1].set_title('True Mask', fontsize=24)
    ax[0, 2].imshow(mask_pred_new.astype(np.uint8), cmap='gray')
    ax[0, 2].set_title('Pred Mask', fontsize=24)
    fig.suptitle(title, fontsize=24)
    fig.savefig(save_path)


def save_generate_image(img_real, img_tasks, task_mask, img_fake, save_path, title):
    """
        output image for validation (batch size=1)
        :param img:
        :param true_mask:
        :param mask_pred:
        :param save_path:
        :param intensity:
        :return:
        """
    # img_real: [1, 1, nx, ny]
    img_real, img_tasks, task_mask, img_fake = \
        np.array(img_real.view(img_real.shape[2],-1).to('cpu')), \
        np.array(img_tasks.view(img_tasks.shape[2],-1).to('cpu')), \
        np.array(task_mask.view(task_mask.shape[2], -1).to('cpu')), \
        np.array(img_fake.view(img_fake.shape[2], -1).to('cpu'))

    fig, ax = plt.subplots(1, 4, figsize=(32, 8))
    ax = np.atleast_2d(ax)

    img_real = (img_real + 1) / 2 * 255
    img_tasks = (img_tasks + 1) / 2 * 255
    task_mask = (task_mask + 1) / 2 * 255
    img_fake = (img_fake + 1) / 2 * 255

    ax[0, 0].imshow(img_real.astype(np.uint8), cmap='gray')
    ax[0, 0].set_title('Real Image', fontsize=24)
    ax[0, 1].imshow(img_tasks.astype(np.uint8), cmap='gray')
    ax[0, 1].set_title('Training Image', fontsize=24)
    ax[0, 2].imshow(task_mask.astype(np.uint8), cmap='gray')
    ax[0, 2].set_title('Image Mask', fontsize=24)
    ax[0, 3].imshow(img_fake.astype(np.uint8), cmap='gray')
    ax[0, 3].set_title('Fake Image', fontsize=24)
    fig.suptitle(title, fontsize=24)
    fig.savefig(save_path)


def save_decoder_image(img, save_path):
    # batch_Imgs: [nbatch, channel=n, nx, ny]
    batch_Imgs = np.array(img.to('cpu'))
    nbatch = batch_Imgs.shape[0]
    nchannel = batch_Imgs.shape[1]
    nx = batch_Imgs.shape[2]
    nrow = math.ceil(nchannel ** 0.5)
    fig, ax = plt.subplots(nrow, nrow, figsize=(nx * nrow // 32, nx * nrow // 32),
                           sharey=True, sharex=True)
    ax = np.atleast_2d(ax)
    for i in range(nbatch):
        for j in range(nchannel):
            ax = plt.subplot(nrow, nrow, j + 1)
            img = np.array(batch_Imgs[i,j])
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            plt.imshow(img.astype(np.uint8), cmap='gray')
            plt.axis('off')
            fig.tight_layout()
        fig.savefig(save_path + '_batch%03d.png' % i)


def SS_tasks(image, mode='ccc', seed=None, **kwargs):

    """
    :param image:
    :param mode:'cp_center_crop = ccc', 'cp_patches_crop = cpc_values',
                'cp_rdregion_crop = crc_values', 'swap_patches = sp_values'
    :param seed:
    :param kwargs:
    :return:
    """

    mode = mode.lower()

    image = (image - np.min(image)) / np.max(image)
    if seed is not None:
        np.random.seed(seed=seed)

    allowedtypes = {
        'ccc': 'ccc_values',
        'cpc': 'cpc_values',
        'crc': 'crc_values',
        'sp': 'sp_values',
        'mmr': 'tir_values',
        'amb': 'tir_values'}

    kwdefaults = {
        # cp_center_crop
        'crop_size_center': [50, 50],

        # cp_patches_crop
        'npatch': 10,
        'crop_size_patches': [16, 16],

        # cp_rdregion_crop
        'thresh': 3,
        'prec': 25.,

        'paint_style': 1,
        'mean': 0.,
        'var': 0.1,

        # swap_patches
        'nswap': 10,

        # two_image_reconstruction
        'corres_img': np.zeros((128, 128)),
        'npatch_16': 4
    }

    allowedkwargs = {
        'ccc_values': ['crop_size_center', 'paint_style', 'mean', 'var'],
        'cpc_values': ['npatch', 'crop_size_patches', 'paint_style', 'mean', 'var'],
        'crc_values': ['thresh', 'prec', 'paint_style', 'mean', 'var'],
        'sp_values': ['nswap', 'crop_size_patches'],
        'tir_values': ['corres_img', 'npatch_16']}

    for key in kwargs:
        if key not in allowedkwargs[allowedtypes[mode]]:
            raise ValueError('%s keyword not in allowed keywords %s' %
                             (key, allowedkwargs[allowedtypes[mode]]))

    # Set kwarg defaults
    for kw in allowedkwargs[allowedtypes[mode]]:
        kwargs.setdefault(kw, kwdefaults[kw])

    if mode in ["ccc", "cpc", "crc"]:

        if mode == "ccc":
            mask = cp_center_crop(image, kwargs['crop_size_center'])
        elif mode == "cpc":
            mask = cp_patches_crop(image, kwargs['npatch'], kwargs['crop_size_patches'])
        elif mode == "crc":
            mask = cp_rdregion_crop(image, kwargs['thresh'], kwargs['prec'])

        # painting with intensity 0, or with random noise, or add noise to images
        if kwargs['paint_style'] == 0:
            style = np.random.randint(1, 4, size=1)

        if kwargs['paint_style'] == 1 or style == 1:
            ss_img = image * mask
        elif kwargs['paint_style'] == 2 or style == 2:
            noise = np.random.random(image.shape)
            ss_img = image * mask + noise * (1 - mask)
        elif kwargs['paint_style'] == 3 or style == 3:
            noise = random_noise(image, mode='gaussian',
                                 mean=kwargs['mean'], var=kwargs['var'])
            ss_img = image * mask + noise * (1 - mask)
    elif mode == "sp":
        ss_img, mask = swap_patches(image, kwargs['nswap'], kwargs['crop_size_patches'])
    elif mode == "mmr" or mode == 'amb':
        ss_img, mask = two_image_reconstruction(image, kwargs['corres_img'], kwargs['npatch_16'])

    return ss_img, mask, mode


def cp_center_crop(img, crop_size):
    '''
    context prediction ( crop a centering mask with size [px, py])

    :param img:
    :param crop_size:
    :param style:
        0: random choose style form 1,2,3
        1: painting with intensity 0
        2: painting with random noise
        3: add noise to images
    :return:
    '''
    px, py = crop_size
    ix, iy = img.shape
    mask = np.ones(img.shape)
    mask[(ix - px) // 2: (ix + px) // 2, (iy - py) // 2: (iy + py) // 2] = 0

    return(mask)


def cp_patches_crop(img, npatch, crop_size):
    """
    context prediction ( crop N multi-patches with size [px, py])

    :param img:
    :param crop_size:
    :return:
    """
    px, py = crop_size
    (ix, iy) = img.shape
    mask = np.ones(img.shape)
    # region of the patch center
    cx, cy = np.random.randint(0, ix + 1, npatch), np.random.randint(0, iy + 1, npatch)
    for i in range(npatch):
        # region boundary
        rx_1, rx_2, ry_1, ry_2 = max(cx[i] - px // 2, 0), min(cx[i] + px // 2, ix), \
                                 max(cy[i] - py // 2, 0), min(cy[i] + py // 2, iy)
        mask[rx_1: rx_2, ry_1: ry_2] = 0

    return (mask)


def cp_rdregion_crop(img, thresh, prec):
    """

    :param img:
    :param thresh:
    :param prec:
    :return:
    """
    # context prediction ( crop random region with proportion perc%)

    (ix, iy) = img.shape
    mask = np.ones(img.shape)
    connects = [[-1, -1], [0, -1], [1, -1], [1, 0],
                [1, 1], [0, 1], [-1, 1], [-1, 0]]

    # init seed [sx, sy]
    sx, sy = np.random.randint(0, ix, 3), np.random.randint(0, iy, 3)
    seedList = [[i,j] for i,j in zip(sx, sy)]

    while (len(seedList) > 0 and np.sum(mask) >= ix * iy * (1 - prec / 100)):
        currentPoint = seedList.pop(0)
        mask[currentPoint[0], currentPoint[1]] = 0
        for i in range(8):
            tmpX = currentPoint[0] + connects[i][0]
            tmpY = currentPoint[1] + connects[i][1]
            if tmpX < 0 or tmpY < 0 or tmpX >= ix or tmpY >= iy:
                continue
            if np.random.randint(0,10,1) < thresh and mask[tmpX, tmpY] == 1:
                mask[tmpX, tmpY] = 0
                seedList.append([tmpX, tmpY])
    return(mask)


def swap_patches(img, nswap, crop_size):
    """
    # reconstruction ( swap N multi-patches with size [px, py])
    :param
    img:
    :param
    crop_size:
    :return:
    """
    px, py = crop_size
    (ix, iy) = img.shape
    mask = np.ones(img.shape)
    # region of the patch center
    cx_1, cy_1, cx_2, cy_2 = np.random.randint(px // 2, ix - px // 2, nswap), \
                             np.random.randint(py // 2, iy - py // 2, nswap), \
                             np.random.randint(px // 2, ix - px // 2, nswap), \
                             np.random.randint(py // 2, iy - py // 2, nswap)
    for i in range(nswap):
        # region boundary
        temp = img[cx_1[i] - px // 2: cx_1[i] + px // 2, cy_1[i] - px // 2: cy_1[i] + px // 2]
        img[cx_1[i] - px // 2: cx_1[i] + px // 2, cy_1[i] - px // 2: cy_1[i] + px // 2] = \
            img[cx_2[i] - px // 2: cx_2[i] + px // 2, cy_2[i] - px // 2: cy_2[i] + px // 2]
        img[cx_2[i] - px // 2: cx_2[i] + px // 2, cy_2[i] - px // 2: cy_2[i] + px // 2] = temp

        mask[cx_1[i] - px // 2: cx_1[i] + px // 2, cy_1[i] - px // 2: cy_1[i] + px // 2] = 0
        mask[cx_2[i] - px // 2: cx_2[i] + px // 2, cy_2[i] - px // 2: cy_2[i] + px // 2] = 0

    return (img, mask)


def two_image_reconstruction(img, corres_img, npatch):
    """
    reconstruction with two images ( replace by corresponding image, N multi-patches with size [px, py])

    :param img:
    :param corres_img:
    :param crop_size:
    :return:
    """
    (ix, iy) = img.shape
    mask = np.ones(img.shape)
    # region of the patch center
    patches = np.random.randint(0, 16, npatch)
    for patch in patches:
        # region boundary
        rx, ry = patch // 4, patch % 4
        img[int(rx * ix//4): int((rx + 1) * ix//4), int(ry * iy//4): int((ry + 1) * iy//4)] = \
            corres_img[int(rx * ix//4): int((rx + 1) * ix//4), int(ry * iy//4): int((ry + 1) * iy//4)]
        mask[int(rx * ix//4): int((rx + 1) * ix//4), int(ry * iy//4): int((ry + 1) * iy//4)] = 0

    return (img, mask)


def cca(image, ref, device):
    # image: [nbatch=1, channel=4, nx, ny]
    image = np.array(image.to('cpu'))
    ref = np.array(ref.to('cpu'))
    image_new = np.zeros(image.shape)

    image_new[0, 0] = image[0, 0]
    for slice in range(1, image.shape[1]):
        if np.sum(image[0, slice]) != 0:
            thres = np.sum(ref[0, slice]) * 0.3
            nb_components, output, stats, centroids = \
                cv2.connectedComponentsWithStats(image[0, slice].astype(np.uint8),
                                                 connectivity=4)
            sizes = stats[:, -1]

            img2 = np.zeros(output.shape)

            for i in range(1, nb_components):
                if sizes[i] > thres:
                    img2[output == i] = 1
            image_new[0, slice] = img2

        else:
            image_new[0, slice] = image[0, slice]

    image_new = torch.from_numpy(image_new)
    return (image_new.to(device=device, dtype=torch.float32))


def img_rotation(img, rotation):
    if rotation == 0:
        return img
    elif rotation == 90:
        return list(map(list, zip(*img[::-1])))
    elif rotation == 180:
        return list([i[::-1] for i in img[::-1]])
    else:
        return list(map(list, zip(*img)))[::-1]

