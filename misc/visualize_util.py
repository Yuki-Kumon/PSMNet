# -*- coding: utf-8 -*-

"""
visualizeの時に使う。
Author :
    Yuki Kumon
Last Update :
    2019-11-06
"""


import sys
sys.path.append('.')

from misc.File_util import File_util

import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
import japanize_matplotlib


class visualize_util():
    '''
    used for visualization
    '''

    def __init__(self):
        pass

    def visualize(self, image_list, annot_list, preds_list, root=None, transform=True):
        '''
        3枚の画像をまとめてプロット
        '''
        # もし、画像の書き出し用の変換がまだならここで行う
        if transform:
            image_list = self.image_trans(image_list)
            annot_list = self.annot_trans(annot_list)
            preds_list = self.annot_trans(preds_list)

        # 書き出し
        for i in range(len(image_list)):
            image_list[i] = 255 - image_list[i]
            out_image = np.concatenate([image_list[i], annot_list[i], preds_list[i]], axis=1)
            if root:
                cv2.imwrite(os.path.join(root, 'visualize_{}.png').format(i), out_image)
            else:
                cv2.imwrite('./saved/visualize_{}.png'.format(i), out_image)

    def respectively_visualize(self, image_list, annot_list, preds_list, output_index, root=None, transform=True):
        '''
        output indexに合わせて画像を書き出す
        '''
        # もし、画像の書き出し用の変換がまだならここで行う
        if transform:
            image_list = self.image_trans(image_list)
            annot_list = self.annot_trans(annot_list)
            preds_list = self.annot_trans(preds_list)

        # 書き出し
        for i in range(len(output_index)):
            image_list[i] = 255 - image_list[i]
            # バラバラに書き出す
            if root:
                cv2.imwrite(os.path.join(root, 'visualize_origin_{}.png'.format(output_index[i])), image_list[output_index[i]])
                cv2.imwrite(os.path.join(root, 'visualize_annotation_{}.png'.format(output_index[i])), annot_list[output_index[i]])
                cv2.imwrite(os.path.join(root, 'visualize_prediction_{}.png'.format(output_index[i])), preds_list[output_index[i]])
            else:
                cv2.imwrite(os.path.join('./output', 'visualize_origin_{}.png'.format(output_index[i])), image_list[output_index[i]])
                cv2.imwrite(os.path.join('./output', 'visualize_annotation_{}.png'.format(output_index[i])), annot_list[output_index[i]])
                cv2.imwrite(os.path.join('./output', 'visualize_prediction_{}.png'.format(output_index[i])), preds_list[output_index[i]])

    def visualize_plt(self, image_list, annot_list, preds_list, output_index, root=None, transform=True):
        '''
        3枚の画像をまとめてプロット
        matplotlibを用いる
        '''
        # もし、画像の書き出し用の変換がまだならここで行う
        if transform:
            image_list = self.image_trans(image_list)
            annot_list = self.annot_trans(annot_list)
            preds_list = self.annot_trans(preds_list)

        # 書き出し
        for i in range(len(output_index)):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(10, 4.5), dpi=200)
            ax1.imshow(image_list[output_index[i]], cmap="gray", vmin=0, vmax=255)
            ax1.set_title('入力画像')
            ax2.imshow(255 - annot_list[output_index[i]], cmap="gray", vmin=0, vmax=1)
            ax2.set_title('正解ラベル')
            ax3.imshow(255 - preds_list[output_index[i]], cmap="gray", vmin=0, vmax=1)
            ax3.set_title('推定ラベル')
            if root:
                fig.savefig(os.path.join(root, 'visualize_plt_{}.png'.format(output_index[i])))
            else:
                fig.savefig(os.path.join('./output', 'visualize_plt_{}.png'.format(output_index[i])))

    def visualize_on_name(self, image_list, names_list, trans_mode='prediction', root=None):
        '''
        あとで画像を結合できるよう、指定のルールに従って書き出す
        '''
        if root:
            # ディレクトリがないなら作成
            File_util.create_folder(root)
        if trans_mode == 'prediction':
            image_list = self.annot_trans(image_list)
            names_list = [x + '_prediction.png' for x in names_list]
        elif trans_mode == 'annotation':
            image_list = self.annot_trans(image_list)
            names_list = [x + '_annotation.png' for x in names_list]
        elif trans_mode == 'original':
            image_list = [255 - x for x in self.image_trans(image_list)]
            names_list = [x + '_original.png' for x in names_list]
        else:
            print('input valid trans_mode string! yours: {}'.format(trans_mode))
            sys.exit()

        # 書き出し
        for i in range(len(image_list)):
            cv2.imwrite(os.path.join(root, names_list[i]), image_list[i])

    @staticmethod
    def image_trans(image_list):
        '''
        書き出し用に画像を255倍する
        '''
        for i in range(len(image_list)):
            image_list[i] = (image_list[i] * 255).astype(np.uint8)

        return image_list

    @staticmethod
    def annot_trans(annot_list):
        for i in range(len(annot_list)):
            annot_list[i] = ((1 - annot_list[i]) * 255).astype(np.uint8)

        return annot_list
