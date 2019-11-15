# -*- coding: utf-8 -*-

"""
trim image to create sub images
Author :
    Yuki Kumon
Last Update :
    2019-11-15
"""

import sys
sys.path.append('.')

from misc.dependences.attr_dict import AttributeDict
from misc.dependences.File_util import File_util

import yaml
import os
import numpy as np
import cv2


with open('./configs/configs.yml', 'r') as f:
    config = AttributeDict(yaml.load(f, Loader=yaml.SafeLoader))

# set path
original_root = config.fields()['path']['original']
edit_root = config.fields()['path']['edit']

# make folders
util = File_util()
util.create_folder(os.path.join(original_root, 'band3s'))
util.create_folder(os.path.join(original_root, 'band3bs'))
util.create_folder(os.path.join(original_root, 'depth'))

# load image size
original_size = config.fields()['size']['original']
trimed_size = config.fields()['size']['trim']
stride = config.fields()['size']['stride']

stride = [int(x) for x in stride]
length = [int(np.floor((original_size[i] - trimed_size[i]) / stride[i])) for i in range(2)]


# trim and save
def trim(img, target_root, img_name, ext='.tif'):
    # load original images and annotations
    img_sub = []
    name_list = []

    for j in range(length[1]):
        for i in range(length[0]):
            img_sub.append(img[
                stride[0] * i:stride[0] * i + trimed_size[0],
                stride[1] * j:stride[1] * j + trimed_size[1]
            ])
            name_list.append(os.path.splitext(os.path.split(img_name)[1])[0] + '_' + str(i) + '_' + str(j))

    # save
    save_name_list = []
    for i in range(len(img_sub)):
        cv2.imwrite(os.path.join(target_root, name_list[i] + ext), img_sub[i])
        save_name_list.append(os.path.join(target_root, name_list[i] + ext))
    return save_name_list


# execute
band3s_img = cv2.imread(os.path.join(original_root, 'band3s.tif'), cv2.IMREAD_GRAYSCALE)[9:-9, 9:-9]
band3bs_img = cv2.imread(os.path.join(original_root, 'band3bs.tif'), cv2.IMREAD_GRAYSCALE)[9:-9, 9:-9]
depth_img = np.load(os.path.join(original_root, 'img_dis.npy'))[0, 9:-9, 9:-9]

band3s_list = trim(band3s_img, os.path.join(edit_root, 'band3s'), 'band3s.tif')
band3bs_list = trim(band3bs_img, os.path.join(original_root, 'band3bs'), 'band3bs.tif')
depth_list = trim(depth_img, os.path.join(original_root, 'depth'), 'depth.png', ext='.png')


# write csv
