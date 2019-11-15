# -*- coding: utf-8 -*-

"""
Train
Author :
    Yuki Kumon
Last Update :
    2019-11-15
"""


import sys

from tqdm import tqdm  # epoch内でプログレスバーを表示(keras的に)
import numpy as np
import yaml
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from absl import app, flags, logging
from absl.flags import FLAGS

from models.PSMnet import PSMNet
from dataloader.AsterLoader import AsterLoader
from misc.File_util import File_util


flags.DEFINE_integer('epoch', 50, 'epoch number')
flags.DEFINE_integer('batch_size', 4, 'batch size')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_bool('is_cuda', False, 'whether cuda is used or not')
flags.DEFINE_bool('pre_trained', False, 'whether model is pretrained or not')
flags.DEFINE_string('optimizer', 'Adam', 'select optimizer')
flags.DEFINE_string('criterion', 'SmoothL1Loss', 'select criterion')
flags.DEFINE_float('validation_rate', 'validation number rate when spliting dataset')
flags.DEFINE_string('tensor_board_log_dir', './tensorboard_log', 'tensorboardX logging folder')
flags.DEFINE_string('config_path', './configs/configs.yml', 'config file path')
flags.DEFINE_string('csv_path', './dataset/edit/result.csv', 'csv path for dataloader')
flags.DEFINE_string('save_path', './model.tar', 'model save path')
flags.DEFINE_float('maxdisp', 120.0, 'max disparity')


def main(_argv):

    TENSOR_BOARD_LOG_DIR = FLAGS.tensor_board_log_dir
    writer = SummaryWriter(TENSOR_BOARD_LOG_DIR)
    logging.info('set tensorboardX writer')

    # load model
    psmnet = PSMNet(FLAGS.maxdisp)

    # load optimizer
    if FLAGS.optimizer == 'Adam':
        optimizer = optim.Adam(psmnet.parameters(), lr=FLAGS.lr)
        logging.info('Adam is loaded as optimizer')
    else:
        logging.info('invalid optimizer is input!: {}'.format(FLAGS.optimizer))
        sys.exit()

    # load criterion
    # criterion
    if FLAGS.criterion == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss()
        logging.info('SmoothL1Loss is loaded as segmentation criterion.')
    else:
        logging.info('invalid criterion is input!: {}'.format(FLAGS.criterion))
        sys.exit()

    # convert for cuda
    if FLAGS.is_cuda:
        psmnet = psmnet.to('cuda')
        criterion = criterion.to('cuda')
        logging.info('use cuda')
    else:
        logging.info('NOT use cuda')

    File = File_util()

    # set dataloader
    train_loader, val_loader = AsterLoader(FLAGS.csv_path)

    # load checkpoint
    if FLAGS.pre_trained:
        psmnet, optimizer, epoch_old = File.load_checkpoint(FLAGS.save_path, psmnet, optimizer)
        logging.info('load checkpoint')
    else:
        epoch_old = 0
        logging.info('NOT load checkpoint')

    # train
    for epoch in range(epoch_old + 1, epoch_old + FLAGS.epoch + 1):
        model, optimizer, criterion = train(FLAGS.epoch + epoch_old, epoch, FLAGS.batch_size, train_loader, psmnet, optimizer, criterion, writer, FLAGS.is_cuda)
        File.save_model(epoch, psmnet, optimizer, FLAGS.save_path)
    # test
    model, optimizer, criterion = test(FLAGS.batch_size, val_loader, psmnet, optimizer, criterion, FLAGS.is_cuda)
    # save
    File.save_model(epoch, psmnet, optimizer, FLAGS.save_path)


def train(max_epoch, epoch, batch, loader, model, optimizer, criterion, writer, is_cuda):
    '''
    training function
    '''
    model.train()

    epoch_losses = []
    epoch_str = str(epoch) + '/' + str(max_epoch)

    with tqdm(total=len(loader) * batch) as pbar:
        for i, data in enumerate(loader):
            if i == 0:
                pbar.set_postfix(OrderedDict(epoch=epoch_str, loss=0.0))
            else:
                pbar.set_postfix(OrderedDict(epoch=epoch_str, loss=np.mean(epoch_losses)))
            # train
            optimizer.zero_grad()

            left_img = data['left']
            right_img = data['right']
            target_disp = data['disp']

            mask = (target_disp > 0)
            mask = mask.detach_()

            if is_cuda:
                left_img = left_img.to('cuda')
                right_img = right_img.to('cuda')

            disp1, disp2, disp3 = model(left_img, right_img)
            loss1, loss2, loss3 = criterion(disp1[mask], disp2[mask], disp3[mask], target_disp[mask])
            total_loss = 0.5 * loss1 + 0.7 * loss2 + 1.0 * loss3

            if is_cuda:
                epoch_losses.append(total_loss.to('cpu').data)
            else:
                epoch_losses.append(total_loss.data)
            total_loss.backward()
            optimizer.step()

            # プログレスバーを進める
            pbar.update(batch)

    writer.add_scalar("loss", np.mean(epoch_losses), epoch)

    return model, optimizer, criterion


def test(batch, loader, model, optimizer, criterion, is_cuda):
    '''
    test function
    '''
    model.eval()

    epoch_losses = []

    with tqdm(total=len(loader) * batch) as pbar:
        for i, data in enumerate(loader):
            if i == 0:
                pbar.set_postfix(OrderedDict(epoch='validation', loss=0.0))
            else:
                pbar.set_postfix(OrderedDict(epoch='validation', loss=np.mean(epoch_losses)))
            # train
            optimizer.zero_grad()

            left_img = data['left']
            right_img = data['right']
            target_disp = data['disp']

            mask = (target_disp > 0)
            mask = mask.detach_()

            if is_cuda:
                left_img = left_img.to('cuda')
                right_img = right_img.to('cuda')

            disp1, disp2, disp3 = model(left_img, right_img)
            loss1, loss2, loss3 = criterion(disp1[mask], disp2[mask], disp3[mask], target_disp[mask])
            total_loss = 0.5 * loss1 + 0.7 * loss2 + 1.0 * loss3

            if is_cuda:
                epoch_losses.append(total_loss.to('cpu').data)
            else:
                epoch_losses.append(total_loss.data)
            total_loss.backward()
            optimizer.step()

            # プログレスバーを進める
            pbar.update(batch)

    return model, optimizer, criterion


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
