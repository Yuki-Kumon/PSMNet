# -*- coding: utf-8 -*-i

"""
Image triming tool
Author :
    Yuki Kumon
Last Update :
    2019-11-02
"""


import os
import numpy as np
import cv2
import glob
import json
from tqdm import trange

import sys
sys.path.append('.')
from misc.File_util import File_util


class ImageTrimTool():
    '''
    学習用に画像をトリムする
    端の欠損値は無視する
    スリックが多めに含まれているもののみ使うことにする
    validation用の画像のトリムにも対応
    '''

    def __init__(self, original_size, trimed_size, stride=[64, 64], tp_ratio=0.03):
        self.original_size = original_size[0], original_size[1]
        self.trimed_size = trimed_size[0], trimed_size[1]
        self.stride = stride
        self.tp_ratio = tp_ratio
        # self.threshold = True

        # loop length
        self.len = [int(np.floor((self.original_size[i] - self.trimed_size[i]) / self.stride[i])) for i in range(2)]

        # holder
        self.img_sub = []
        self.annot_sub = []
        self.name_list = []

        # json output
        self.path_list = []

    def _load(self, img_path, annot_path):
        '''
        load image and resize it
        '''
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        annot = cv2.imread(annot_path, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, self.original_size), cv2.resize(annot, self.original_size)

    def _subimg(self, img, annot, img_name):
        '''
        cut images and annotations
        '''
        # load original images and annotations
        img_sub = []
        annot_sub = []
        name_list = []

        for j in range(self.len[1]):
            for i in range(self.len[0]):
                img_sub.append(img[
                    self.stride[0] * i:self.stride[0] * i + self.trimed_size[0],
                    self.stride[1] * j:self.stride[1] * j + self.trimed_size[1]
                ])
                annot_sub.append(annot[
                    self.stride[0] * i:self.stride[0] * i + self.trimed_size[0],
                    self.stride[1] * j:self.stride[1] * j + self.trimed_size[1]
                ])
                name_list.append(os.path.splitext(os.path.split(img_name)[1])[0] + '_' + str(i) + '_' + str(j))
                # 画像の端の黒いところ対策
                if(np.amin(img_sub[-1]) == 0):
                    del img_sub[-1]
                    del annot_sub[-1]
                    del name_list[-1]

        # return img_sub, annot_sub
        # threshold
        if self.threshold:
            img_sub, annot_sub, name_list = self._threshold(img_sub, annot_sub, name_list)
        self.img_sub += img_sub
        self.annot_sub += annot_sub
        self.name_list += name_list
        if name_list:
            self.path_list.append(os.path.split(img_name)[1])

    def _threshold(self, img_sub, annot_sub, name_list):
        '''
        TP比が一定以下の画像を落とす
        先に_subimg()のループを全て回してから実行する
        '''
        # holder
        img_sub_out = []
        annot_sub_out = []
        name_list_out = []

        for i in range(len(img_sub)):
            # check TP ratio
            positive_map = np.where(annot_sub[i] == 0, 1, 0)
            if (np.mean(positive_map) >= self.tp_ratio):
                img_sub_out.append(img_sub[i])
                annot_sub_out.append(annot_sub[i])
                name_list_out.append(name_list[i])
        return img_sub_out, annot_sub_out, name_list_out

    def _compute_subimg_list(self, image_root):
        '''
        subimageを計算し、リストに格納する
        _submig()のループを計算する
        '''
        file_list = glob.glob(os.path.join(image_root, '*.tif'))
        annot_list = [os.path.splitext(file_list[i])[0] + '_shape.png' for i in range(len(file_list))]
        # compute subimg
        for i in trange(len(file_list), desc='compute subimages'):
            img, annot = self._load(file_list[i], annot_list[i])
            self._subimg(img, annot, file_list[i])
        """
        # threshold
        if self.threshold:
            self._threshold()
        """

    def _save(self, target_root):
        '''
        計算したsubimgを元に画像を保存する
        '''
        # フォルダを作成
        File_util.create_folder(target_root)
        for i in trange(len(self.img_sub), desc='save images'):
            cv2.imwrite(os.path.join(target_root, self.name_list[i] + '.tif'), self.img_sub[i])
            cv2.imwrite(os.path.join(target_root, self.name_list[i] + '_shape.png'), self.annot_sub[i])
        # save json
        file = open(os.path.join(target_root, 'result.json'), 'w')
        json.dump(self.path_list, file)

    def __call__(self, input_path, target_path, threshold=True):
        self.threshold = threshold
        self._compute_subimg_list(input_path)
        self._save(target_path)

    def concat(self, input_path, json_path, target_path, rule='*_prediction.png', save_name='{}_prediction_concat.png'):
        '''
        1枚のtif画像に結合する
        欠損しているところは白抜きで補完する
        重なっているところは平均値を取る？→多数決で決定することにした
        '''
        # フォルダを作成
        File_util.create_folder(target_path)
        # 計算すべきファイルのリストを取得
        with open(json_path) as f:
            name_list = json.load(f)

        # 各オリジナルファイルごとに処理していく
        for name in name_list:
            # 結果を入力する配列
            concat_result = np.zeros(self.original_size, dtype='int')
            plus_num = np.zeros_like(concat_result, dtype='int')
            # 対応するファイルのリストを取得
            file_list = glob.glob(os.path.join(
                input_path,
                os.path.splitext(name)[0] + rule
            ))

            for file_name in file_list:
                # 分割された画像を取得
                img_here = self._slick_to_label(cv2.imread(file_name, cv2.IMREAD_GRAYSCALE))
                # 元画像での位置を取得
                index = [int(i) for i in file_name.split('_')[-3:-1]]
                # 埋め込む
                concat_result[
                    self.stride[0] * index[0]:self.stride[0] * index[0] + self.trimed_size[0],
                    self.stride[1] * index[1]:self.stride[1] * index[1] + self.trimed_size[1]
                ] += img_here
                plus_num[
                    self.stride[0] * index[0]:self.stride[0] * index[0] + self.trimed_size[0],
                    self.stride[1] * index[1]:self.stride[1] * index[1] + self.trimed_size[1]
                ] += 1
            # 重なりの処理
            plus_num = np.where(plus_num == 0, 1, plus_num)
            concat_result = (concat_result / plus_num)
            # 疑わしいところは全てoilslickあり
            # concat_result = np.where(concat_result < 255, 0, 255).astype('int8')
            concat_result = self._label_to_slick(concat_result)
            # 書き出す
            cv2.imwrite(os.path.join(
                target_path,
                save_name.format(os.path.splitext(name)[0])
            ), concat_result)

    @staticmethod
    def _slick_to_label(img):
        return np.where(img == 0, 1, 0)

    @staticmethod
    def _label_to_slick(img):
        return np.where(img.astype('int') == 1, 0, 255).astype('int')


if __name__ == '__main__':
    """
    sanity check
    """

    # img = (np.random.rand(512, 512) * 255 + 1).astype('int')
    # annot = np.where(np.random.rand(512, 512) > 0.9, 0, 255)

    # cls = ImageTrimTool((6179, 5886), (512, 512))
    # cls('./data/oilslick/training', './data/_edit/slick')

    cls = ImageTrimTool((6179, 5886), (512, 512))
    cls.concat('./data/_edit/slick', './data/_edit/slick/result.json', './data/hoge/')

    # cls._compute_subimg_list('./data/oilslick/training')
    # cls._save('./data/_edit/slick')
