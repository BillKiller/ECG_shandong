# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 19:47

@ author: javis
'''

import os
import copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal
from tqdm import trange
import pickle


def count_labels(data, file2idx):
    '''
    统计每个类别的样本数
    :param data:
    :param file2idx:
    :return:
    '''
    cc = [0] * config.num_classes
    for fp in data:
        for i in file2idx[fp]:
            cc[i] += 1
    return np.array(cc)


def resample(sig, target_point_num=None):
    '''
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    '''
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig


def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(
        loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def verflip(sig):
    '''
    信号竖直翻转
    :param sig:
    :return:
    '''
    return sig[::-1, :]


def shift(sig, interval=20):
    '''
    上下平移
    :param sig:
    :return:
    '''
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset
    return sig


def transform(sig, train=False):
    # 前置不可或缺的步骤
    sig = resample(sig, config.target_point_num)
    # # 数据增强
    if train:
        if np.random.randn() > 0.5:
            sig = scaling(sig)
        if np.random.randn() > 0.5:
            sig = verflip(sig)
        if np.random.randn() > 0.5:
            sig = shift(sig)
    # 后置不可或缺的步骤
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, data_path, train=True):
        super(ECGDataset, self).__init__()
        self.train = train
        data_path = data_path + '/train{}.csv'.format(config.kfold) if train else data_path + "/dev{}.csv".format(config.kfold)
        self.dd = pd.read_csv(data_path, sep=',')
        self.data = self.dd['id']
        X_train_value = []
        if os.path.exists(data_path + str(config.target_point_num) + '.pkl'):
            with open(data_path + str(config.target_point_num) + '.pkl', 'rb') as f:
                self.X = pickle.load(f)
        else:
            for index in trange(len(self.data)):
                fid = self.data[index]+'.csv'
                file_path = os.path.join(config.train_dir, fid)
                df = pd.read_csv(file_path, sep=',').values
                x = transform(df, self.train)
                X_train_value.append(x)
            self.X = X_train_value
            with open(data_path + str(config.target_point_num) + '.pkl', 'wb') as f:
                pickle.dump(self.X, f)
        Y_train_value = []
        for item in self.dd['label']:
            Y_train_value.append([int(i) for i in item.split(',')])
        self.labels = np.array(Y_train_value)  # TODO can faster
        self.wc  = 1. / np.log(np.sum(self.labels, axis = 0))

    def __getitem__(self, index):
        x = self.X[index]
        target = self.labels[index]
        target = torch.tensor(target, dtype=torch.float32)
        return x, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    d = ECGDataset(config.train_data)
    # print(d[0])
