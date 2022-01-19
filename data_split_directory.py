import os

import shutil

import pandas as pd
from tqdm import tqdm

def copy2dir(dataset_type):
    os.makedirs(dataset_type, exist_ok=True)
    df = pd.read_csv('./data/ecg_data/{}.csv'.format(dataset_type))
    file_name = df['id']
    for fid in tqdm(file_name,desc="Runing {dataset_type}"):
        shutil.copy('./data/ecg_data/{}.csv'.format(fid),
                    '{}/{}.csv'.format(dataset_type, fid))


for name in ['train', 'dev', 'test']:
    copy2dir(name)


#%%
import torch

from torch import nn

from sklearn.model_selection import StratifiedKFold

import pandas as pd


df = pd.read_csv('./label_and_example/train_label_1217.csv')

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k , (trn_idx, val_idx) in enumerate(kfold.split(df['id'].values, df['label'].values)):
    fold_train_data = df.iloc[trn_idx]
    fold_val_data = df.iloc[val_idx]
    fold_train_data.to_csv("train_{}.csv".format(k))
    fold_val_data.to_csv("dev_{}.csv".format(k))
