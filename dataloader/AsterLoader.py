# -*- coding: utf-8 -*-

"""
DataLoader
Author :
    Yuki Kumon
Last Update :
    2019-11-15
"""


from torch.utils.data import Dataset
from torchvision import transforms
import torch.utils.data  # データセット読み込み関連
import torch.utils.data.random_split  # データセット分割

from PIL import Image
import numpy as np

import random
import csv


class AsterDataset(Dataset):
    '''
    Aster衛星画像で計算した標高画像を読み込む
    Datasetクラスを継承
    '''

    def __init__(self, csv_path, trans1=None, trans2=None, trans3=None):
        self.trans1 = trans1
        self.trans2 = trans2
        self.trans3 = trans3
        # read csv
        with open(csv_path) as f:
            reader = csv.reader(f)
            self.path_list = [x for x in reader]

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        data = {}
        im_list = []

        # load images
        im_list.append(Image.open(self.path_list[idx][0]))
        im_list.append(Image.open(self.path_list[idx][1]))
        im_list.append(Image.open(self.path_list[idx][2]))

        # transform
        if self.trans1:
            im_list[0] = self.trans1(im_list[0])
            im_list[1] = self.trans1(im_list[1])
        if self.trans2:
            im_list[2] = self.trans2(im_list[2])
        if self.trans3:
            im_list = self.trans3(im_list)

        data['left'] = im_list[0]
        data['right'] = im_list[1]
        data['disp'] = im_list[2]

        del im_list

        return data


class Flip_Segmentation(object):
    """
    Flip images and labels randomly
    """
    def __init__(self):
        pass

    def __call__(self, arr_list):
        image1_arr, image2_arr, GT_arr = arr_list
        GT_arr = GT_arr[None]

        # tensorをnumpyに変換しておく
        image1_arr = image1_arr.numpy()
        image2_arr = image2_arr.numpy()
        GT_arr = GT_arr.numpy()

        if random.choices([True, False]):
            image1_arr = np.flip(image1_arr, 0).copy()
            image2_arr = np.flip(image2_arr, 0).copy()
            GT_arr = np.flip(GT_arr, 0).copy()

        if random.choices([True, False]):
            image1_arr = np.flip(image1_arr, 1).copy()
            image2_arr = np.flip(image2_arr, 1).copy()
            GT_arr = np.flip(GT_arr, 1).copy()
        # tensorに変換して返す
        image1_arr = torch.from_numpy(image1_arr)
        image2_arr = torch.from_numpy(image2_arr)
        GT_arr = torch.from_numpy(GT_arr)

        arr_list = [image1_arr, image2_arr, GT_arr[0]]

        return arr_list


class Rotate_Segmentation(object):
    """
    Rotate images and labels randomly
    """
    def __init__(self):
        pass

    def __call__(self, arr_list):
        image1_arr, image2_arr, GT_arr = arr_list

        # tensorをnumpyに変換しておく
        image1_arr = image1_arr.numpy()
        image2_arr = image2_arr.numpy()
        GT_arr = GT_arr.numpy()[None]
        # print(GT_arr.shape)

        n = random.choices([0, 1, 2, 3])
        image1_arr = np.rot90(image1_arr, n[0], (-2, -1)).copy()
        image2_arr = np.rot90(image2_arr, n[0], (-2, -1)).copy()
        GT_arr = np.rot90(GT_arr, n[0], (-2, -1)).copy()
        # tensorに変換して返す
        image1_arr = torch.from_numpy(image1_arr)
        image2_arr = torch.from_numpy(image2_arr)
        GT_arr = torch.from_numpy(GT_arr)

        arr_list = [image1_arr, image2_arr, GT_arr[0]]

        return arr_list


class Normalize_depending_on_image(object):
    """
    Normalize using average and std of image
    """

    def __init__(self):
        pass

    def __call__(self, image):
        mean = torch.mean(image)
        std = torch.std(image)

        # 平均0.5、標準偏差0.5の画像にしてreturnする
        return ((image - mean) / std + 1.) / 2.


def set_trans(image_size):
    trans1 = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
        Normalize_depending_on_image()
    ])
    trans2 = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    trans3 = transforms.Compose([
        Flip_Segmentation(),
        Rotate_Segmentation()
    ])
    return trans1, trans2, trans3


def AsterLoader(csv_path, batch_size=4, split=True, val_rate=0.1, shuffle=True):
    trans1, trans2, trans3 = set_trans()
    dataset = AsterDataset(csv_path, trans1, trans2, trans3)
    if split:
        # split dataset randomly
        n_sample = len(dataset)
        val_size = int(n_sample * val_rate)
        train_size = n_sample - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        return (
            torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
            torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        )
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
