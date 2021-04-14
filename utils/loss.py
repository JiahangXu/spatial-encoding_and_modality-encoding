from __future__ import print_function, division, absolute_import, unicode_literals

import torch.nn as nn
import numpy as np
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 0.0001

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


def hausdorff_compute(pred, groundtruth, spacing):
    pred = np.squeeze(pred)
    groundtruth = np.squeeze(groundtruth)

    ITKPred = sitk.GetImageFromArray(pred, isVector=False)
    ITKPred.SetSpacing(spacing)
    ITKTrue = sitk.GetImageFromArray(groundtruth, isVector=False)
    ITKTrue.SetSpacing(spacing)
    try:
        pred_surface = sitk.LabelContour(ITKPred)
        gt_surface = sitk.LabelContour(ITKTrue)

        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

        # Hausdorff distance
        hausdorff_distance_filter.Execute(pred_surface, gt_surface)

        surface_distance_results= hausdorff_distance_filter.GetHausdorffDistance()

        return surface_distance_results
    except:
        return -1




def multi_hausdorff_distance(batch_imgs, batch_gts, spacing=(1.25,1.25)):
    # batch_Imgs: [nbatch=1, channel=4, nx, ny]
    gt, img = np.array(batch_gts.to('cpu')), \
              np.array(batch_imgs.to('cpu'))

    HD_list = []

    # --------------
    # LV endo
    img_1, gt_1 = np.zeros(img.shape[2:]), np.zeros(gt.shape[2:])
    img_1[img[0, 2] == 1] = 1
    gt_1[gt[0, 2] == 1] = 1

    if np.sum(gt_1) == 0:
        HD_list.append(-1)
    elif np.sum(img_1) == 0:
        HD_list.append(64)
    else:
        HD_list.append(hausdorff_compute(np.array(img_1, dtype=int),
                                         np.array(gt_1, dtype=int),
                                         spacing=spacing))

    # --------------
    # LV epi
    img_2, gt_2 = np.zeros(img.shape[2:]), np.zeros(gt.shape[2:])
    img_2[(img[0, 1] == 1) | (img[0, 2] == 1)] = 1
    gt_2[(gt[0, 1] == 1) | (gt[0, 2] == 1)] = 1

    if np.sum(gt_2) == 0:
        HD_list.append(-1)
    elif np.sum(img_2) == 0:
        HD_list.append(64)
    else:
        HD_list.append(hausdorff_compute(np.array(img_2, dtype=int),
                                         np.array(gt_2, dtype=int),
                                         spacing=spacing))

    # RV endo
    img_3, gt_3 = np.zeros(img.shape[2:]), np.zeros(gt.shape[2:])
    img_3[img[0, 3] == 1] = 1
    gt_3[gt[0, 3] == 1] = 1

    if np.sum(gt_3) == 0:
        HD_list.append(-1)
    elif np.sum(img_3) == 0:
        HD_list.append(64)
    else:

        HD_list.append(hausdorff_compute(np.array(img_3, dtype=int),
                                         np.array(gt_3, dtype=int),
                                         spacing=spacing))

    return HD_list
