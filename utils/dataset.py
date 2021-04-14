from __future__ import print_function, division, absolute_import, unicode_literals
import os
import glob
import numpy as np
import nibabel as nib
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch import randperm
from torch._utils import _accumulate
from utils.utils import SS_tasks, img_rotation
import pandas as pd
import random
from itertools import permutations


def get_slice(data_dir_list, output_dir, output_name, is_mmr = False, is_amb = False,
             is_intercept = 0, data_aug = 'all', check_lab = False):
    """
    :param data_dir: "/xxx/xxx/*.nii.gz"
    :param output_dir: the dictionary to store dict
    :param output_name: output name
    :param is_intercept: in some case, the top and bottom slice need to be removed, as they are lack of information
    :param cut: if cut is False, then output all slices; if cut is a number x,
        then output the first x slices in the whole dataset.
    :param lab: if the processing data is image label.
    """
    dict_byIndex = []
    nifti_index = 0
    index = 0
    img_num = 0

    if is_mmr:
        for [data_dir1, data_type1, _], [data_dir2, data_type2, _] in permutations(data_dir_list, 2):
            print(data_type1, data_type2)

            # data_dir1 = os.path.join(data_dir1, "*.nii.gz")
            # all_files = glob.glob(data_dir1)
            all_files = []

            train_list = [3, 5, 6, 8, 9, 10, 19, 26, 29, 33, 34, 37, 38, 40, 44]
            for id in train_list:
                all_files.append(os.path.join(data_dir1, 'patient{}_{}.nii.gz'.format(id, data_type1)))
            img_num += len(all_files)
            print(len(all_files))

            for img_loc in all_files:
                nifti_index += 1

                # load main image and its label
                img_dir = os.path.join(*img_loc.split(sep="/")[:-1])
                img_name = img_loc.split(sep="/")[-1]
                img = nib.load(os.path.join(img_dir, img_name))
                img_aff = img.affine
                img = np.array(img.get_fdata(), np.float32)

                lab_dir = img_dir.replace('img', 'lab')
                lab_name = img_name.replace(".nii.gz", "_manual.nii.gz")
                lab = nib.load(os.path.join(lab_dir, lab_name))
                lab = np.array(lab.get_fdata(), np.int32)

                n_slice = img.shape[2] + 1
                thres1, thres2 = round(n_slice / 3), round(n_slice / 3 * 2)
                center_x = round((min(np.where(lab > 0)[0]) + max(np.where(lab > 0)[0])) / 2)
                center_y = round((min(np.where(lab > 0)[1]) + max(np.where(lab > 0)[1])) / 2)

                # load corresponding image and its label
                corres_name = img_name.replace(data_type1, data_type2)
                corres_img = nib.load(os.path.join(data_dir2, corres_name))
                corres_aff = corres_img.affine
                # corres_img = np.array(corres_img.get_fdata(), np.int32)

                corres_lab_dir = data_dir2.replace('img', 'lab')
                corres_lab_name = corres_name.replace(".nii.gz", "_manual.nii.gz")
                corres_lab = nib.load(os.path.join(corres_lab_dir, corres_lab_name))
                corres_lab = np.array(corres_lab.get_fdata(), np.int32)

                corres_cx = round((min(np.where(corres_lab > 0)[0]) +
                                   max(np.where(corres_lab > 0)[0])) / 2)
                corres_cy = round((min(np.where(corres_lab > 0)[1]) +
                                   max(np.where(corres_lab > 0)[1])) / 2)

                for slice_index in range(0 + is_intercept, img.shape[2] - is_intercept):

                    position = 'apical' if slice_index < thres1 else \
                        'middle' if slice_index < thres2 else 'basal'

                    img_z = np.dot(img_aff, [center_x, center_y, slice_index, 1])[2]

                    dis, corres_index = 1000, 0
                    for i in range(corres_img.shape[2]):
                        corres_z = np.dot(corres_aff, [corres_cx, corres_cy, i, 1])[2]
                        if abs(corres_z - img_z) < dis:
                            dis = abs(corres_z - img_z)
                            corres_index = i

                    if data_aug == 'all':
                        rotation = [0, 90, 180, 270]
                    else:
                        rotation = [0]
                    for i in rotation:
                        temp_dict = {'index': index,
                                     'nifti_index': nifti_index,
                                     'img_type': data_type1,
                                     'img_dir': img_dir,
                                     'img_name': img_name,
                                     'slice_index': slice_index,
                                     'slice_position': position,
                                     'center_x': center_x,
                                     'center_y': center_y,
                                     'corres_dir': data_dir2,
                                     'corres_name': corres_name,
                                     'corres_slice_index': corres_index,
                                     'corres_cx': corres_cx,
                                     'corres_cy': corres_cy,
                                     'rotation': i}
                        dict_byIndex.append(temp_dict)
                        index += 1
            print(index)

    elif is_amb:
        for data_dir, data_type, cen_from_lab in data_dir_list:
            # data_dir1 = os.path.join(data_dir1, "*.nii.gz")
            # all_files = glob.glob(data_dir1)
            all_files = []

            train_list = [3, 5, 6, 8, 9, 10, 19, 26, 29, 33, 34, 37, 38, 40, 44]
            for id in train_list:
                all_files.append(os.path.join(data_dir, 'patient{}_{}.nii.gz'.format(id, data_type)))
            img_num += len(all_files)
            print(len(all_files))

            for img_loc in all_files:
                nifti_index += 1

                img_dir = os.path.join(*img_loc.split(sep="/")[:-1])
                img_name = img_loc.split(sep="/")[-1]
                img = nib.load(os.path.join(img_dir, img_name))
                img = np.array(img.get_fdata(), np.float32)

                lab_dir = img_dir.replace('img', 'lab')
                lab_name = img_name.replace(".nii.gz", "_manual.nii.gz")
                lab = nib.load(os.path.join(lab_dir, lab_name))
                lab = np.array(lab.get_fdata(), np.int32)

                n_slice = img.shape[2] + 1
                thres1, thres2 = round(n_slice / 3), round(n_slice / 3 * 2)
                center_x = round((min(np.where(lab > 0)[0]) + max(np.where(lab > 0)[0])) / 2)
                center_y = round((min(np.where(lab > 0)[1]) + max(np.where(lab > 0)[1])) / 2)

                for slice_index in range(0 + is_intercept, img.shape[2] - is_intercept):
                    if check_lab and np.sum(lab[:, :, slice_index]) == 0:
                        continue

                    position = 'apical' if slice_index < thres1 else \
                        'middle' if slice_index < thres2 else 'basal'

                    if data_aug == 'all':
                        rotation = [0, 90, 180, 270]
                    elif data_aug == 'LGE_T2' and (data_type == "LGE" or data_type == "T2"):
                        rotation = [0, 90, 180, 270]
                    else:
                        rotation = [0]
                    for i in rotation:
                        for j in range(2):
                            temp_dict = {'index': index,
                                         'nifti_index': nifti_index,
                                         'img_type': data_type,
                                         'img_dir': img_dir,
                                         'img_name': img_name,
                                         'slice_index': slice_index,
                                         'slice_position': position,
                                         'center_x': center_x,
                                         'center_y': center_y,
                                         'rotation': i}
                            dict_byIndex.append(temp_dict)
                            index += 1
            print(index)

    else:
        for data_dir, data_type, cen_from_lab in data_dir_list:
            if cen_from_lab:
                data_dir = os.path.join(data_dir, "*.nii.gz")
                all_files = glob.glob(data_dir)

                for img_loc in all_files:
                    nifti_index += 1

                    img_dir = os.path.join(*img_loc.split(sep="/")[:-1])
                    img_name = img_loc.split(sep="/")[-1]
                    img = nib.load(os.path.join(img_dir, img_name))
                    img = np.array(img.get_fdata(), np.float32)

                    lab_dir = img_dir.replace('img', 'lab')
                    lab_name = img_name.replace(".nii.gz", "_manual.nii.gz")
                    lab = nib.load(os.path.join(lab_dir, lab_name))
                    lab = np.array(lab.get_fdata(), np.int32)

                    n_slice = img.shape[2] + 1
                    thres1, thres2 = round(n_slice / 3), round(n_slice / 3 * 2)
                    center_x = round((min(np.where(lab > 0)[0]) + max(np.where(lab > 0)[0])) / 2)
                    center_y = round((min(np.where(lab > 0)[1]) + max(np.where(lab > 0)[1])) / 2)

                    for slice_index in range(0 + is_intercept, img.shape[2] - is_intercept):
                        if check_lab and np.sum(lab[:, :, slice_index]) == 0:
                            continue

                        position = 'apical' if slice_index < thres1 else \
                            'middle' if slice_index < thres2 else 'basal'

                        if data_aug == 'all':
                            rotation = [0, 90, 180, 270]
                        elif data_aug == 'LGE_T2' and (data_type == "LGE" or data_type == "T2"):
                            rotation = [0, 90, 180, 270]
                        else:
                            rotation = [0]
                        for i in rotation:
                            temp_dict = {'index': index,
                                         'nifti_index': nifti_index,
                                         'img_type': data_type,
                                         'img_dir': img_dir,
                                         'img_name': img_name,
                                         'slice_index': slice_index,
                                         'slice_position': position,
                                         'center_x': center_x,
                                         'center_y': center_y,
                                         'rotation': i}
                            dict_byIndex.append(temp_dict)
                            index += 1

            else:
                csv_file = pd.read_csv(os.path.join(data_dir, "center_point.csv"))
                all_files = [os.path.join(data_dir, i) for i in csv_file['name']]

                for i, img_loc in enumerate(all_files):
                    nifti_index += 1

                    img_dir = os.path.join(*img_loc.split(sep="/")[:-1])
                    img_name = img_loc.split(sep="/")[-1]
                    img = nib.load(os.path.join(img_dir, img_name))
                    img = np.array(img.get_fdata(), np.float32)

                    n_slice = img.shape[2] + 1
                    thres1, thres2 = round(n_slice / 3), round(n_slice / 3 * 2)
                    center_x = csv_file.iloc[i]['cx'] - 1
                    center_y = csv_file.iloc[i]['cy'] - 1

                    for slice_index in range(0 + is_intercept, img.shape[2] - is_intercept):

                        position = 'apical' if slice_index < thres1 else \
                            'middle' if slice_index < thres2 else 'basal'

                        if data_aug == 'all':
                            rotation = [0, 90, 180, 270]
                        elif data_aug == 'LGE_T2' and (data_type == "LGE" or data_type == "T2"):
                            rotation = [0, 90, 180, 270]
                        else:
                            rotation = [0]

                        for j in rotation:
                            temp_dict = {'index': index,
                                         'nifti_index': nifti_index,
                                         'img_type': data_type,
                                         'img_dir': img_dir,
                                         'img_name': img_name,
                                         'slice_index': slice_index,
                                         'slice_position': position,
                                         'center_x': center_x,
                                         'center_y': center_y,
                                         'rotation': j}
                            dict_byIndex.append(temp_dict)
                            index += 1

            img_num += len(all_files)

    np.save(os.path.join(output_dir, output_name), dict_byIndex)
    dict_byIndexDF = pd.DataFrame(dict_byIndex)
    dict_byIndexDF.to_csv(os.path.join(output_dir, 'test.csv'))
    print('Get [%d] slices from [%d] NiFTI files.' % (len(dict_byIndex), img_num))
    return(dict_byIndex)


def data_pre_process(data_dir_list, output_dir, output_name, is_intercept):
    '''
    :param search_dir:
    :param output_dir:
    :param output_name:

    # for unlabeled image
    # 对于有label的图像，将label中心点存下来；对于没有label的图像，选csv中的中心点记录下来；
    # LGE_T2_aug表示是否对LGE和T2 imgae 进行旋转
    dict_byIndex = get_slice([[os.path.join(os.path.pardir, "Dataset", "Cine_EDES"), "bSSFP", False],
                              [os.path.join(os.path.pardir, "Dataset", "ACDC", "Unlabeled"), "bSSFP", False],
                              [os.path.join(os.path.pardir, "Dataset", "ACDC", "Labeled", "img"), "bSSFP", True],
                              [os.path.join(os.path.pardir, "Dataset", "MSCMRSeg", "bSSFP", "img"), "bSSFP", True],
                              [os.path.join(os.path.pardir, "Dataset", "MSCMRSeg", "LGE", "img"), "LGE", True],
                              [os.path.join(os.path.pardir, "Dataset", "MSCMRSeg", "T2", "img"), "T2", True]],
                             os.path.join(os.path.pardir, "Dataset"), "all_Dict_byIndex.npy",
                             is_intercept=0, data_aug = 'LGE_T2')

    # for mmr task:
    dict_byIndex = get_slice([[os.path.join(os.path.pardir, "Dataset", "MSCMRSeg", "bSSFP", "img"), "bSSFP", True],
                          [os.path.join(os.path.pardir, "Dataset", "MSCMRSeg", "LGE", "img"), "LGE", True],
                          [os.path.join(os.path.pardir, "Dataset", "MSCMRSeg", "T2", "img"), "T2", True]],
                         os.path.join(os.path.pardir, "Dataset"), "mmr_task_Dict_byIndex_ntrain15.npy",
                         is_mmr=True, data_aug = 'all', is_intercept=0, check_lab=True)
    # for spatial encoding task:
    dict_byIndex = get_slice([[os.path.join(os.path.pardir, "Dataset", "MSCMRSeg", "bSSFP", "img"), "bSSFP", True],
                              [os.path.join(os.path.pardir, "Dataset", "MSCMRSeg", "LGE", "img"), "LGE", True],
                              [os.path.join(os.path.pardir, "Dataset", "MSCMRSeg", "T2", "img"), "T2", True]],
                             os.path.join(os.path.pardir, "Dataset"), "amb_task_Dict_byIndex_ntrain15.npy",
                             is_amb=True, is_intercept=0, data_aug = 'all')

    # for segmentation image
    dict_byIndex = get_slice([[os.path.join(os.path.pardir, "Dataset", "MSCMRSeg", "bSSFP", "img"), "bSSFP", True]],
                             os.path.join(os.path.pardir, "Dataset"), "labeled_bSSFP_Dict_byIndex.npy",
                             data_aug = 'all', is_intercept=0, check_lab=True)

    dict_byIndex = get_slice([[os.path.join(os.path.pardir, "Dataset", "MSCMRSeg", "LGE", "img"), "LGE", True]],
                             os.path.join(os.path.pardir, "Dataset"), "labeled_LGE_Dict_byIndex.npy",
                             data_aug = 'all', is_intercept=0, check_lab=True)

    dict_byIndex = get_slice([[os.path.join(os.path.pardir, "Dataset", "MSCMRSeg", "T2", "img"), "T2", True]],
                             os.path.join(os.path.pardir, "Dataset"), "labeled_T2_Dict_byIndex.npy",
                             data_aug = 'all', is_intercept=0, check_lab=True)

    # dict_byIndex = get_slice([[os.path.join(os.path.pardir, "Dataset", "MSCMRSeg", "bSSFP", "img"), "bSSFP", True],
    #                           [os.path.join(os.path.pardir, "Dataset", "MSCMRSeg", "LGE", "img"), "LGE", True],
    #                           [os.path.join(os.path.pardir, "Dataset", "MSCMRSeg", "T2", "img"), "T2", True]],
    #                          os.path.join(os.path.pardir, "Dataset"), "labeled_Dict_byIndex.npy",
    #                          data_aug = 'all', is_intercept=0, check_lab=True)

    :return:
    '''
    dict_byIndex = get_slice(data_dir_list, output_dir, output_name,
                            is_intercept=is_intercept)
    return(dict_byIndex)


class UnlabeledDataLoader2D(Dataset):
    """
    load 2D silce from 3D NiFTI images. Every calling returns a 2D slice.
    """
    def __init__(self, dictForSlice_dir, img_size, task_style=0):
        """
        :param dictForSlice_dir: the pre-processed file for image dictionary, refer to the function "data_pre_process"
        :param img_size: the size [nx, ny] of the 2D slice;
        """
        self.dict_forSlice = pd.DataFrame(list(
            np.load(dictForSlice_dir, allow_pickle=True)))
        self.img_size = img_size
        self.task_style = task_style

        print("Number of files used: %s" % len(self.dict_forSlice))

    def __getitem__(self, file_idx):
        """
        :param file_idx:
        :return: [1, nx, ny]
        """
        img_info = self.dict_forSlice.iloc[file_idx]
        img_dir = img_info['img_dir']
        img_name = img_info['img_name']
        slice_index = img_info['slice_index']
        nifti_index = img_info['nifti_index']
        slice_position = img_info['slice_position']
        center_x = img_info['center_x']
        center_y = img_info['center_y']
        rotation = img_info['rotation']

        img = nib.load(os.path.join(img_dir, img_name))
        img = np.array(img.get_fdata(), np.float32)[:, :, slice_index]

        # padding or cropping
        img_new = img[int(center_x - self.img_size // 2): int(center_x + self.img_size // 2),
                      int(center_y - self.img_size // 2): int(center_y + self.img_size // 2)]
        img_new = img_rotation(img_new, rotation)

        # normalization the intensity to [0, 1] for self-supervised task
        img_new = (img_new - np.min(img_new)) / np.max(img_new)

        if self.task_style == 'map':
            tempDF = self.dict_forSlice.loc[(self.dict_forSlice['nifti_index'] == nifti_index) &
                                            (self.dict_forSlice['slice_position'] != slice_position) &
                                            (self.dict_forSlice['slice_index'] != slice_index)]
            corres_slice_index = random.sample(list(tempDF.index), 1)

            corres_info = self.dict_forSlice.iloc[corres_slice_index[0]]

            corres_img = nib.load(os.path.join(corres_info['img_dir'], corres_info['img_name']))
            corres_img = np.array(corres_img.get_fdata(), np.float32)[:, :, corres_info['slice_index']]

            # padding or cropping
            corres_img_new = corres_img[int(center_x - self.img_size // 2): int(center_x + self.img_size // 2),
                             int(center_y - self.img_size // 2): int(center_y + self.img_size // 2)]
            corres_img_new = img_rotation(corres_img_new, rotation)
            corres_img_new = (corres_img_new - np.min(corres_img_new)) / np.max(corres_img_new)

            SS_img, SS_mask, mode = SS_tasks(img_new, corres_img=corres_img_new, mode=self.task_style)

        if self.task_style == 'map1':
            tempDF = self.dict_forSlice.loc[(self.dict_forSlice['nifti_index'] == nifti_index) &
                                            (abs(self.dict_forSlice['slice_index'] - slice_index) == 1)]
            corres_slice_index = random.sample(list(tempDF.index), 1)

            corres_info = self.dict_forSlice.iloc[corres_slice_index[0]]

            corres_img = nib.load(os.path.join(corres_info['img_dir'], corres_info['img_name']))
            corres_img = np.array(corres_img.get_fdata(), np.float32)[:, :, corres_info['slice_index']]

            # padding or cropping
            corres_img_new = corres_img[int(center_x - self.img_size // 2): int(center_x + self.img_size // 2),
                             int(center_y - self.img_size // 2): int(center_y + self.img_size // 2)]
            corres_img_new = img_rotation(corres_img_new, rotation)
            corres_img_new = (corres_img_new - np.min(corres_img_new)) / np.max(corres_img_new)

            SS_img, SS_mask, mode = SS_tasks(img_new, corres_img=corres_img_new, mode='map')

        elif self.task_style == 'map2':
            tempDF = self.dict_forSlice.loc[(self.dict_forSlice['nifti_index'] == nifti_index) &
                                            (abs(self.dict_forSlice['slice_index'] - slice_index) <= 2)]
            corres_slice_index = random.sample(list(tempDF.index), 1)

            corres_info = self.dict_forSlice.iloc[corres_slice_index[0]]

            corres_img = nib.load(os.path.join(corres_info['img_dir'], corres_info['img_name']))
            corres_img = np.array(corres_img.get_fdata(), np.float32)[:, :, corres_info['slice_index']]

            # padding or cropping
            corres_img_new = corres_img[int(center_x - self.img_size // 2): int(center_x + self.img_size // 2),
                             int(center_y - self.img_size // 2): int(center_y + self.img_size // 2)]
            corres_img_new = img_rotation(corres_img_new, rotation)
            corres_img_new = (corres_img_new - np.min(corres_img_new)) / np.max(corres_img_new)

            SS_img, SS_mask, mode = SS_tasks(img_new, corres_img=corres_img_new, mode='map')


        elif self.task_style == 'mmr':
            corres_img = nib.load(os.path.join(img_info['corres_dir'], img_info['corres_name']))
            corres_img = np.array(corres_img.get_fdata(), np.float32)[:, :, img_info['corres_slice_index']]
            corres_cx, corres_cy = img_info['corres_cx'], img_info['corres_cy']

            # padding or cropping
            corres_img_new = corres_img[int(corres_cx - self.img_size // 2): int(corres_cx + self.img_size // 2),
                             int(corres_cy - self.img_size // 2): int(corres_cy + self.img_size // 2)]
            corres_img_new = img_rotation(corres_img_new, rotation)
            corres_img_new = (corres_img_new - np.min(corres_img_new)) / np.max(corres_img_new)

            SS_img, SS_mask, mode = SS_tasks(img_new, corres_img=corres_img_new, mode=self.task_style)

        else:
            SS_img, SS_mask, mode = SS_tasks(img_new, self.task_style)

        # nomalization
        img_real = self._img_norm(img_new)
        img_tasks = self._img_norm(SS_img)
        task_mask = torch.tensor(SS_mask).unsqueeze(0)

        return {'img_real': img_real,
                'img_tasks': img_tasks,
                'task_mask': task_mask,
                'task_mode': mode}

    def __len__(self):
        return len(self.dict_forSlice)

    def _img_norm(self, img):
        '''
        :param img: [nx, ny]
        :return: tensor data [1, nx, ny] in [-1, 1]
        '''
        img_norm = (img - np.min(img)) / np.max(img) * 2 - 1
        img_norm = torch.tensor(img_norm)
        img_norm = img_norm.unsqueeze(0)
        return img_norm


class MetaUnlabeledDataLoader2D(Dataset):
    """
    load 2D silce from 3D NiFTI images. Every calling returns a 2D sets.

    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, img_size=128, episz=1000, k_support=15, k_query=75,
                 task_list=['ccc', 'cpc', 'sp', 'mmr', 'amb', 'amb2']):
        """
        :param dictForSlice_dir: the pre-processed file for image dictionary, refer to the function "data_pre_process"
        :param img_size: a numbr indicates the size nx = ny of the squared 2D slice, [nx, ny];
        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param episz: episode size of sets, not of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of query imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """
        self.dict_forSlice = pd.DataFrame(list(np.load("../Dataset/all_Dict_byIndex.npy", allow_pickle=True)))
        self.dict_forSlice_mmr = pd.DataFrame(list(np.load("../Dataset/mmr_task_Dict_byIndex_new.npy", allow_pickle=True)))
        self.img_size = img_size
        self.task_list = task_list
        self.episz = episz

        print('Dataset Intialized: Episode size:%d, %d-support, %d-query' % (episz, k_support, k_query))

    def _load_img_with_task(self, img_idx, task):
        """
        :param file_idx:
        :return: [1, nx, ny]
        """
        if task == 'mmr':
            img_info = self.dict_forSlice_mmr.iloc[img_idx]
        else:
            img_info = self.dict_forSlice.iloc[img_idx]

        img_dir = img_info['img_dir']
        img_name = img_info['img_name']
        slice_index = img_info['slice_index']
        nifti_index = img_info['nifti_index']
        slice_position = img_info['slice_position']
        center_x = img_info['center_x']
        center_y = img_info['center_y']
        rotation = img_info['rotation']

        img = nib.load(os.path.join(img_dir, img_name))
        img = np.array(img.get_fdata(), np.float32)[:, :, slice_index]

        # padding or cropping
        img_new = img[int(center_x - self.img_size // 2): int(center_x + self.img_size // 2),
                  int(center_y - self.img_size // 2): int(center_y + self.img_size // 2)]
        img_new = img_rotation(img_new, rotation)

        # normalization the intensity to [0, 1] for self-supervised task
        img_new = (img_new - np.min(img_new)) / np.max(img_new)

        if task in ['amb', 'amb1', 'amb2', 'mmr']:
            if task in ['amb', 'amb1', 'amb2']:
                if task == 'amb':
                    tempDF = self.dict_forSlice.loc[(self.dict_forSlice['nifti_index'] == nifti_index) &
                                                    (self.dict_forSlice['slice_position'] != slice_position) &
                                                    (self.dict_forSlice['slice_index'] != slice_index)]
                else:
                    thres = 1 if task == 'amb1' else 2
                    task = 'amb'
                    tempDF = self.dict_forSlice.loc[(self.dict_forSlice['nifti_index'] == nifti_index) &
                                                    (abs(self.dict_forSlice['slice_index'] - slice_index) <= thres)]

                corres_slice_index = random.sample(list(tempDF.index), 1)
                corres_info = self.dict_forSlice.iloc[corres_slice_index[0]]

                corres_img = nib.load(os.path.join(corres_info['img_dir'], corres_info['img_name']))
                corres_img = np.array(corres_img.get_fdata(), np.float32)[:, :, corres_info['slice_index']]

                # padding or cropping
                corres_img_new = corres_img[int(center_x - self.img_size // 2): int(center_x + self.img_size // 2),
                                 int(center_y - self.img_size // 2): int(center_y + self.img_size // 2)]
                corres_img_new = img_rotation(corres_img_new, rotation)
                corres_img_new = (corres_img_new - np.min(corres_img_new)) / np.max(corres_img_new)

            else:
                corres_img = nib.load(os.path.join(img_info['corres_dir'], img_info['corres_name']))
                corres_img = np.array(corres_img.get_fdata(), np.float32)[:, :, img_info['corres_slice_index']]
                corres_cx, corres_cy = img_info['corres_cx'], img_info['corres_cy']

                # padding or cropping
                corres_img_new = corres_img[int(corres_cx - self.img_size // 2): int(corres_cx + self.img_size // 2),
                                 int(corres_cy - self.img_size // 2): int(corres_cy + self.img_size // 2)]
                corres_img_new = img_rotation(corres_img_new, rotation)
                corres_img_new = (corres_img_new - np.min(corres_img_new)) / np.max(corres_img_new)

            SS_img, SS_mask, mode = SS_tasks(img_new, corres_img=corres_img_new, mode=task)

        else:
            SS_img, SS_mask, mode = SS_tasks(img_new, mode=task)

        # nomalization
        img_real = self._img_norm(img_new)
        img_tasks = self._img_norm(SS_img)
        task_mask = torch.tensor(SS_mask).unsqueeze(0)

        return img_tasks, img_real, task_mask

    def _img_norm(self, img):
        '''
        :param img: [nx, ny]
        :return: tensor data [1, nx, ny] in [-1, 1]
        '''
        img_norm = (img - np.min(img)) / np.max(img) * 2 - 1
        img_norm = torch.tensor(img_norm,dtype=torch.float32)
        img_norm = img_norm.unsqueeze(0)
        return img_norm

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= episz-1
        :param index:
        :return:
        """

        if 'mmr' in self.task_list:
            if index < len(self.dict_forSlice):
                cur_task = random.sample([i for i in self.task_list if i != 'mmr'], 1)[0]
                img_task, img_real, _ = self._load_img_with_task(index, cur_task)
            else:
                cur_task = 'mmr'
                img_task, img_real, _ = self._load_img_with_task(index - len(self.dict_forSlice), cur_task)

        else:
            cur_task = random.sample(self.task_list, 1)[0]
            img_task, img_real, task_mask = self._load_img_with_task(index, cur_task)
        # print(img_real.shape, img_task.shape, task_mask.shape, cur_task)
        return {'img_real': img_real,
                'img_tasks': img_task,
                # 'task_mask': task_mask,
                'task_mode': cur_task}

    def __len__(self):
        if 'mmr' in self.task_list:
            return len(self.dict_forSlice) + len(self.dict_forSlice_mmr)
        else:
            return len(self.dict_forSlice)


class LabeledDataLoader2D(Dataset):
    """
    load segmentation data, including labeled MR image and its label.
    """
    def __init__(self, dictForSlice_dir, n_class=4, class_intensity=[0, 200, 500, 600],
                 img_size=128, nifti_train=-1, mode='train', nifti_val=5, nifti_test=30,
                 lab=True, one_hot=True, img_mark=".nii.gz", lab_mark="_manual.nii.gz"):
        """
        :param dictForSlice_dir: the dictionary to get slices from one multi-slice image
        :param img_size:
        :param intensity: a list for the intensity of each label
        :param n_class: the class number of the label, including bg 0.
        :param lab: whether label images are needed. True: need; Flase: not need
        :param one_hot: whether the label images should be processed to one-hot ()
        :param img_mark:
        :param lab_mark:
        """
        origin_dict = pd.DataFrame(list(np.load(dictForSlice_dir, allow_pickle=True)))

        nifti_index_list = np.unique(origin_dict['nifti_index'])
        random.seed(3399)
        random.shuffle(nifti_index_list)
        if mode=='train':
            self.data_aug = True
            if nifti_train == -1:
                nifti_index = nifti_index_list[:int(len(nifti_index_list) - nifti_val - nifti_test)]
            else:
                nifti_index = nifti_index_list[:nifti_train]
        elif mode=='val':
            self.data_aug = False
            nifti_index = nifti_index_list[int(len(nifti_index_list) - nifti_val - nifti_test):
                                           int(len(nifti_index_list) - nifti_test)]
        else:
            self.data_aug = False
            nifti_index = nifti_index_list[int(len(nifti_index_list) - nifti_test):]

        self.nifti_train = len(nifti_index)
        self.nifti_val = nifti_val
        self.nifti_index_list = nifti_index
        self.dict_forSlice = origin_dict[(origin_dict['nifti_index'].isin(nifti_index))]
        if not self.data_aug:
            self.dict_forSlice = self.dict_forSlice[self.dict_forSlice['rotation']==0]
        self.img_size = img_size
        self.class_intensity = class_intensity
        self.n_class = n_class
        self.img_mark = img_mark
        self.lab_mark = lab_mark
        self.lab = lab
        self.one_hot = one_hot

    def __getitem__(self, file_idx):
        """
        :param file_idx:
        :return: [1, nx, ny]
        """
        img_info = self.dict_forSlice.iloc[file_idx]
        img_dir = img_info['img_dir']
        img_name = img_info['img_name']
        slice_index = img_info['slice_index']
        center_x = img_info['center_x']
        center_y = img_info['center_y']
        rotation = img_info['rotation']
        nifti_index = img_info['nifti_index']

        img = nib.load(os.path.join(img_dir, img_name))
        img = np.array(img.get_fdata(), np.float32)[:, :, slice_index]

        # padding or cropping
        img_new = img[int(center_x - self.img_size // 2): int(center_x + self.img_size // 2),
                  int(center_y - self.img_size // 2): int(center_y + self.img_size // 2)]
        img_new = img_rotation(img_new, rotation)
        if nifti_index in [14, 16] and 'LGE' in img_dir:
            img_new[img_new > np.percentile(img_new, 99)] = np.percentile(img_new, 99)
            img_new[img_new < np.percentile(img_new, 1)] = np.percentile(img_new, 1)

        # normalization the intensity to [0, 1] for self-supervised task
        img_new = (img_new - np.min(img_new)) / np.max(img_new) * 2 - 1
        img_new = torch.tensor(img_new).unsqueeze(0)


        lab_dir = img_dir.replace('img', 'lab')
        lab_name = img_name.replace(self.img_mark, self.lab_mark)
        lab = nib.load(os.path.join(lab_dir, lab_name))
        lab = np.array(lab.get_fdata(), np.float32)[:, :, slice_index]

        # padding or cropping
        lab_new = lab[int(center_x - self.img_size // 2): int(center_x + self.img_size // 2),
                  int(center_y - self.img_size // 2): int(center_y + self.img_size // 2)]
        # print(lab_new)
        # print(lab_new.shape)
        # np.save("origin.npy", lab_new)
        lab_new = np.array(img_rotation(lab_new, rotation))
        # print(lab_new)
        # print(lab_new.shape)
        # np.save("rotated.npy", lab_new)

        if self.one_hot:
            lab_data = self._OneHot_lab(lab_new, self.class_intensity)
            lab_data = torch.tensor(lab_data)
        else:
            lab_data = torch.tensor(lab_new).unsqueeze(0)
        return {'image': img_new, 'mask': lab_data, 'index': img_info['index']}

    def __len__(self):
        return len(self.dict_forSlice)

    def _OneHot_lab(self, lab_data, intensity):
        # return semantic_map -> [n_classes, H, W ]
        semantic_map = np.zeros((len(intensity), self.img_size, self.img_size))
        for i in range(len(intensity)):
            temp = np.where(lab_data == intensity[i], 1, 0)
            semantic_map[i, :, :] = temp
        return torch.tensor(semantic_map)


class testPatient42(Dataset):
    """
    load segmentation data, including labeled MR image and its label.
    """
    def __init__(self, dictForSlice_dir, n_class=4, class_intensity=[0, 200, 500, 600],
                 img_size=128, nifti_train=-1, mode='train', nifti_val=5, nifti_test=30,
                 lab=True, one_hot=True, img_mark=".nii.gz", lab_mark="_manual.nii.gz"):
        """
        :param dictForSlice_dir: the dictionary to get slices from one multi-slice image
        :param img_size:
        :param intensity: a list for the intensity of each label
        :param n_class: the class number of the label, including bg 0.
        :param lab: whether label images are needed. True: need; Flase: not need
        :param one_hot: whether the label images should be processed to one-hot ()
        :param img_mark:
        :param lab_mark:
        """
        origin_dict = pd.DataFrame(list(np.load(dictForSlice_dir, allow_pickle=True)))

        nifti_index_list = np.unique(origin_dict['nifti_index'])
        random.seed(3399)
        random.shuffle(nifti_index_list)
        if mode=='train':
            self.data_aug = True
            if nifti_train == -1:
                nifti_index = nifti_index_list[:int(len(nifti_index_list) - nifti_val - nifti_test)]
            else:
                nifti_index = nifti_index_list[:nifti_train]
        elif mode=='val':
            self.data_aug = False
            nifti_index = nifti_index_list[int(len(nifti_index_list) - nifti_val - nifti_test):
                                           int(len(nifti_index_list) - nifti_test)]
        else:
            self.data_aug = False
            nifti_index = nifti_index_list[int(len(nifti_index_list) - nifti_test):]

        nifti_index = [14, 16]
        self.nifti_train = len(nifti_index)
        self.nifti_val = nifti_val
        self.nifti_index_list = nifti_index
        self.dict_forSlice = origin_dict[(origin_dict['nifti_index'].isin(nifti_index))]
        if not self.data_aug:
            self.dict_forSlice = self.dict_forSlice[self.dict_forSlice['rotation'] == 0]
        self.img_size = img_size
        self.class_intensity = class_intensity
        self.n_class = n_class
        self.img_mark = img_mark
        self.lab_mark = lab_mark
        self.lab = lab
        self.one_hot = one_hot

    def __getitem__(self, file_idx):
        """
        :param file_idx:
        :return: [1, nx, ny]
        """
        img_info = self.dict_forSlice.iloc[file_idx]
        img_dir = img_info['img_dir']
        img_name = img_info['img_name']
        slice_index = img_info['slice_index']
        center_x = img_info['center_x']
        center_y = img_info['center_y']
        rotation = img_info['rotation']

        img = nib.load(os.path.join(img_dir, img_name))
        img = np.array(img.get_fdata(), np.float32)[:, :, slice_index]

        # padding or cropping
        img_new = img[int(center_x - self.img_size // 2): int(center_x + self.img_size // 2),
                  int(center_y - self.img_size // 2): int(center_y + self.img_size // 2)]
        img_new = img_rotation(img_new, rotation)
        img_new[img_new > np.percentile(img_new, 99)] = np.percentile(img_new, 99)
        img_new[img_new < np.percentile(img_new, 1)] = np.percentile(img_new, 1)

        # normalization the intensity to [0, 1] for self-supervised task
        img_new = (img_new - np.min(img_new)) / np.max(img_new) * 2 - 1
        img_new = torch.tensor(img_new).unsqueeze(0)


        lab_dir = img_dir.replace('img', 'lab')
        lab_name = img_name.replace(self.img_mark, self.lab_mark)
        lab = nib.load(os.path.join(lab_dir, lab_name))
        lab = np.array(lab.get_fdata(), np.float32)[:, :, slice_index]

        # padding or cropping
        lab_new = lab[int(center_x - self.img_size // 2): int(center_x + self.img_size // 2),
                  int(center_y - self.img_size // 2): int(center_y + self.img_size // 2)]
        # print(lab_new)
        # print(lab_new.shape)
        # np.save("origin.npy", lab_new)
        lab_new = np.array(img_rotation(lab_new, rotation))
        # print(lab_new)
        # print(lab_new.shape)
        # np.save("rotated.npy", lab_new)

        if self.one_hot:
            lab_data = self._OneHot_lab(lab_new, self.class_intensity)
            lab_data = torch.tensor(lab_data)
        else:
            lab_data = torch.tensor(lab_new).unsqueeze(0)
        return {'image': img_new, 'mask': lab_data, 'index': img_info['index']}

    def __len__(self):
        return len(self.dict_forSlice)

    def _OneHot_lab(self, lab_data, intensity):
        # return semantic_map -> [n_classes, H, W ]
        semantic_map = np.zeros((len(intensity), self.img_size, self.img_size))
        for i in range(len(intensity)):
            temp = np.where(lab_data == intensity[i], 1, 0)
            semantic_map[i, :, :] = temp
        return torch.tensor(semantic_map)