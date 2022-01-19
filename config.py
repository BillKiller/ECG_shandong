# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 18:45

@ author: javis
'''
import os


class Config:
    # for data_process.py
    #root = r'D:\ECG'
    root = r'data'
    train_dir = os.path.join(root, 'ecg_data/')
    # test_dir = os.path.join(root, 'ecg_data/testA')
    # train_label = os.path.join(root, 'hf_round1_label.txt')
    # test_label = os.path.join(root, 'hf_round1_subA.txt')
    # arrythmia = os.path.join(root, 'hf_round1_arrythmia.txt')
    train_data = os.path.join(root, 'ecg_data')

    # for train
    #训练的模型名称
    model_name = 'sit'
    #在第几个epoch进行到下一个state,调整lr
    stage_epoch = [128]
    #训练时的batch大小
    batch_size = 32
    #label的类别数
    num_classes = 18
    #最大训练多少个epoch
    max_epoch = 128
    #目标的采样长度
    target_point_num = 2048 
    #保存模型的文件夹
    ckpt = 'ckpt/'
    #保存提交文件的文件夹
    sub_dir = 'submit'
    #初始的学习率
    lr = 3e-5
    #保存模型当前epoch的权重
    kfold = ""
    current_w = 'current_w.pth'
    #保存最佳的权重 你还愿意
    best_w = 'best_w.pth'
    # 学习率衰减 lr/=lr_decay
    lr_decay = 10

    #for test
    temp_dir=os.path.join(root,'temp')
    # SiT
    patch_size = 8
    dim = 256
    mlp_dim = 1024
    dropout = 0.3
    head_num = 8
    depth = 8
    heads = 8



config = Config()
