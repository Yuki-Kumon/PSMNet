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

from misc.dependencies.image_trim_tool import ImageTrimTool
from misc.dependencies.attr_dict import AttributeDict
from misc.dependencies.File_util import File_util

import yaml
import os


with open('./configs/configs.yml', 'r') as f:
    config = AttributeDict(yaml.load(f, Loader=yaml.SafeLoader))

# set path
original_root = config.fields()['path']['original']
edit_root = config.fields()['path']['edit']
# test_annot_root = config.fields()['path']['test_annot']
# csv_path = config.fields()['path']['csv']
# annot = config.fields()['annotation']

# make folders
util = File_util()
slik_path = util.create_folders_with_annot(edit_root, ['slik'])
val_path = util.create_folders_with_annot(edit_root, ['validation'])
# test_path = util.create_folders_with_annot(edit_root, annot)
# test_annot_path = util.create_folders_with_annot(test_annot_root, annot)

# copy test image using annotation
# util.copy_files_by_csv(util.return_path(original_root, 'test'), csv_path, test_annot_root, annot)

# load image size
original_size = config.fields()['size']['original']
trimed_size = config.fields()['size']['trim']
stride_slick = config.fields()['size']['stride_slick']
stride_validation = config.fields()['size']['stride_validation']
tp_ratio = config.fields()['size']['tp_ratio']

# trim and save
Im = ImageTrimTool(original_size, trimed_size, stride=[int(x) for x in stride_slick], tp_ratio=tp_ratio)
Im(os.path.join(original_root, 'training'), slik_path[0])
print('slick finished')
# Im = ImageTrimTool(original_size, trimed_size, stride=[int(trimed_size[i] / 2) for i in range(2)])
Im = ImageTrimTool(original_size, trimed_size, stride=[int(x) for x in stride_validation])
Im(os.path.join(original_root, 'validation'), val_path[0], threshold=False)
print('validation finished')
