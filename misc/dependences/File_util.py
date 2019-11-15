
# -*- coding: utf-8 -*-

"""
File handling tool
Author :
    Yuki Kumon
Last Update :
    2019-06-11
"""


import shutil
import csv
# import yaml
import os
import torch


class File_util():

    def __init__(self):
        pass

    @staticmethod
    def create_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @ staticmethod
    def return_path(root, target):
        return os.path.join(root, target)

    def return_path_list(self, root, target):
        return [self.return_path(root, x) for x in target]

    def create_folders_with_annot(self, root, annot):
        path_list = [self.return_path(root, x) for x in annot]
        for path in path_list:
            self.create_folder(path)
        return path_list

    def copy_files_by_csv(self, original_folder, csv_path, save_root, annot_list):
        with open(csv_path, encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                path = row[0]
                annot = row[1].split(', ')
                for key in annot_list:
                    if key in annot:
                        shutil.copyfile(os.path.join(original_folder, path), os.path.join(self.return_path(save_root, key), path))

    @staticmethod
    def get_option(train, test):
        return {'train': train, 'test': test}

    @classmethod
    def save_model(self, epoch, model, optimizer, save_path):
        # save
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
            save_path
        )
        return 0

    @staticmethod
    def load_checkpoint(path, model, optimizer):
        # load
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        return model, optimizer, epoch


if __name__ == '__main__':
    """
    sanity check
    """
    obj = File_util()
    root = './data/_edit'
    annotation = ['ship', 'wave', 'tide', 'island', 'noise', 'ice']
    obj.create_folders_with_annot(root, annotation)
