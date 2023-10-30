# %%
# compare the crop type classification of RF and SimSiam
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning import loggers as pl_loggers
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import selfsupervised
from selfsupervised.processing import utils

utils.seed_torch()

with open("../../config/croptypes/param_year_invarianz.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

batch_size_pre = cfg["pretraining"]['batch_size']
batch_size_fine = cfg["finetuning"]['batch_size']
lr_fine = cfg["finetuning"]['learning_rate']

# parameter for SimSiam
num_ftrs = cfg["pretraining"]['num_ftrs']
proj_hidden_dim = cfg["pretraining"]['proj_hidden_dim']
pred_hidden_dim = cfg["pretraining"]['pred_hidden_dim']
out_dim = cfg["pretraining"]['out_dim']
lr_pre = cfg["pretraining"]['learning_rate']
epochs_pre = cfg["pretraining"]['epochs']
_gpu = cfg["pretraining"]['gpus']

# %%
# load data for bavaria
# experiment with train/test split for all data
dm_bavaria = selfsupervised.data.croptypes.BavariaDataModule(
    data_dir='../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx',
    batch_size=batch_size_pre,
    num_workers=4,
    experiment='Experiment1')

# data for invariance between crops
dm_crops1 = selfsupervised.data.croptypes.AugmentationExperiments(
    data_dir='../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx',
    batch_size=batch_size_pre,
    num_workers=4,
    experiment='Experiment3')

# %%
# create artificail sin cos dataset with 3 classes


seed = 12345512
np.random.seed(seed)

n = 14
x_data = np.linspace(0, 14, num=n)
y_data = np.cos(x_data)
y_data2 = np.sin(x_data)
y_data3 = np.sin(x_data) + 0.3

# %%
plt.plot(y_data)
plt.plot(y_data2)
plt.plot(y_data3)
# %%

X_train, X_test, y_train, y_test = train_test_split(x_data,
                                                    y_data,
                                                    test_size=0.2,
                                                    random_state=42)

# %%
# setup DataModule
# load data for bavaria
bavaria_data = pd.read_excel(
    "../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx")
bavaria_data = utils.clean_bavarian_labels(bavaria_data)
bavaria_data = utils.remove_false_observation(bavaria_data)
bavaria_data = bavaria_data[(bavaria_data['Date'] >= "03-30")
                            & (bavaria_data['Date'] <= "08-30")]
bavaria_data = utils.rewrite_id_CustomDataSet(bavaria_data)
# %%

bavaria_data

# %%


class TSDataSet(Dataset):
    '''
    :param data: dataset of type pandas.DataFrame
    :param target_col: targeted column name
    :param field_id: name of column with field ids
    :param feature_list: list with target features
    :param callback: preprocessing of dataframe
    '''

    def __init__(self,
                 data,
                 feature_list=[],
                 target_col='NC',
                 field_id='id',
                 time_steps=14,
                 callback=None):
        self.df = data
        self.target_col = target_col
        self.feature_list = feature_list
        self.time_steps = time_steps

        if callback != None:
            self.df = callback(self.df)

        self._fields_amount = len(self.df[field_id].unique())

        # get numpy
        self.y = self.df[self.target_col].values
        self.field_ids = self.df[field_id].values
        self.df = self.df[self.feature_list].values

        if self.y.size == 0:
            print('Target column not in dataframe')
            return
        if self.field_ids.size == 0:
            print('Field id not defined')
            return

        # reshape to 3D
        # field x T x D
        self.df = self.df.reshape(int(self._fields_amount), self.time_steps,
                                  len(self.feature_list))
        self.y = self.y.reshape(int(self._fields_amount), 1, self.time_steps)
        self.field_ids = self.field_ids.reshape(int(self._fields_amount), 1,
                                                self.time_steps)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        x = self.df[idx, :, :]
        y = self.y[idx, 0, 0]

        torchx = self.x2torch(x)
        torchy = self.y2torch(y)
        return torchx, torchy  # , torch.tensor(field_id, dtype=torch.long)

    def x2torch(self, x):
        '''
        return torch for x
        '''
        # nb_obs, nb_features = self.x.shape
        return torch.from_numpy(x).type(torch.FloatTensor)

    def y2torch(self, y):
        '''
        return torch for y
        '''
        return torch.tensor(y, dtype=torch.long)


class Custom1617(Dataset):

    def __init__(self,
                 data,
                 feature_list=[],
                 target_col='NC',
                 field_id='id',
                 time_steps=14,
                 size=0):
        self.df = data
        self.target_col = target_col
        self.feature_list = feature_list
        self.time_steps = time_steps
        self.size = size

        if self.size == 0:
            print('Define data size')
            sys.exit()

        # numpy with augmented data
        # size x 2 x T x D
        self.augmented = np.zeros(
            (self.size, 2, self.time_steps, len(self.feature_list)))
        self.labels = np.zeros((self.size, 1))
        self.sampleData()

    def sampleData(self):
        try:
            for idx in range(self.size):
                ts1, ts2, y = self.get_X1_X2(self.df, self.feature_list)
                self.augmented[idx, 0] = ts1
                self.augmented[idx, 1] = ts2
                self.labels[idx, 0] = y
        except Exception as e:
            print('Error in data generation:', e)
            raise

    def get_X1_X2(self, data, features):
        '''Returns two different timeseries for the same crop
        '''

        # two different years
        year_list = data.Year.unique().tolist()
        random_year1 = random.choice(year_list)
        year_list.remove(random_year1)
        random_year2 = random.choice(year_list)

        # choose same crop but from different years
        field_id1 = random.choice(
            data[(data.Year == random_year1)].id.unique())
        field_id2 = random.choice(
            data[(data.Year == random_year2)].id.unique())

        X1 = data[data.id == field_id1][features].to_numpy()
        X2 = data[data.id == field_id2][features].to_numpy()

        return X1, X2, random_year1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x1 = self.augmented[idx, 0]
        x2 = self.augmented[idx, 1]
        y = self.labels[idx, 0]

        # augmentation based on different years
        aug_x1 = self.x2torch(x1)
        aug_x2 = self.x2torch(x2)
        y = self.y2torch(y)

        # None torch values for x,y
        x = torch.from_numpy(np.array(0)).type(torch.FloatTensor)
        # y = torch.from_numpy(y).type(torch.FloatTensor)

        return (aug_x1, aug_x2), x, y

    def x2torch(self, x):
        '''
        return torch for x
        '''
        # nb_obs, nb_features = self.x.shape
        return torch.from_numpy(x).type(torch.FloatTensor)

    def y2torch(self, y):
        '''
        return torch for y
        '''
        return torch.tensor(y, dtype=torch.long)


class Custom1718(Dataset):

    def __init__(self,
                 data,
                 feature_list=[],
                 target_col='NC',
                 field_id='id',
                 time_steps=14,
                 size=0):
        self.df = data
        self.target_col = target_col
        self.feature_list = feature_list
        self.time_steps = time_steps
        self.size = size

        if self.size == 0:
            print('Define data size')
            sys.exit()

        # numpy with augmented data
        # size x 2 x T x D
        self.augmented = np.zeros(
            (self.size, 2, self.time_steps, len(self.feature_list)))
        self.labels = np.zeros((self.size, 1))
        self.sampleData()

    def sampleData(self):
        try:
            for idx in range(self.size):
                ts1, ts2, y = self.get_X1_X2(self.df, self.feature_list)
                self.augmented[idx, 0] = ts1
                self.augmented[idx, 1] = ts2
                self.labels[idx, 0] = y
        except Exception as e:
            print('Error in data generation:', e)
            raise

    def get_X1_X2(self, data, features):
        '''Returns two different timeseries for the same crop
        '''

        # two different years
        year_list = data.Year.unique().tolist()
        random_year1 = random.choice(year_list)
        year_list.remove(random_year1)

        # choose same crop but from different years
        field_id1 = random.choice(data[(data.Year == 1)].id.unique())
        field_id2 = random.choice(data[(data.Year == 2)].id.unique())

        X1 = data[data.id == field_id1][features].to_numpy()
        X2 = data[data.id == field_id2][features].to_numpy()

        return X1, X2, 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x1 = self.augmented[idx, 0]
        x2 = self.augmented[idx, 1]
        y = self.labels[idx, 0]

        # augmentation based on different years
        aug_x1 = self.x2torch(x1)
        aug_x2 = self.x2torch(x2)
        y = self.y2torch(y)

        # None torch values for x,y
        x = torch.from_numpy(np.array(0)).type(torch.FloatTensor)
        # y = torch.from_numpy(y).type(torch.FloatTensor)

        return (aug_x1, aug_x2), x, y

    def x2torch(self, x):
        '''
        return torch for x
        '''
        # nb_obs, nb_features = self.x.shape
        return torch.from_numpy(x).type(torch.FloatTensor)

    def y2torch(self, y):
        '''
        return torch for y
        '''
        return torch.tensor(y, dtype=torch.long)


bavaria_data["Year"].replace({2016: 0, 2017: 1, 2018: 2}, inplace=True)

# %%
# Vorgehen:
# 1. Pre-Train
# Nimm nur Daten 2016, 2017 und 2018 11 Zeitschritte
# Mach ein Bild von optimal drei Clustern
# Schau ob du eine invariante Darstellung davon bekommst

bavaria_data = bavaria_data[bavaria_data.NC == 4]
bavaria_data
# %%

# %%
# Make test for year classification and
features = [
    'B3_mean', 'B4_mean', 'B5_mean', 'B6_mean', 'B7_mean', 'B8_mean',
    'B8A_mean'
]
train = TSDataSet(bavaria_data, features, 'Year', field_id='id', time_steps=11)
train_dl = DataLoader(train,
                      batch_size=batch_size_pre,
                      shuffle=True,
                      drop_last=False,
                      num_workers=2)
tb_logger2 = pl_loggers.TensorBoardLogger(save_dir="../../logs/")
trainer = pl.Trainer(deterministic=True, max_epochs=100, logger=tb_logger2)
model4 = selfsupervised.model.Attention_LM(
    input_dim=7,
    num_classes=cfg["transformer"]['num_classes'],
    n_head=cfg["transformer"]['n_head'],
    nlayers=cfg["transformer"]['nlayers'],
    batch_size=batch_size_fine,
    lr=lr_pre)
# trainer.fit(model4, train_dl)

# %%
path_to_modeldir = '../../models/yearinvarianz_pretrained_B3-8_100.ckpt'
# torch.save( model4, path_to_modeldir )

# %%
path_to_modeldir = '../../models/yearinvarianz_pretrained_B3-8_100.ckpt'
# model4 = torch.load(path_to_modeldir)
# %%

# %%
transformer = selfsupervised.model.Attention(
    input_dim=7,
    num_classes=cfg["transformer"]['num_classes'],
    n_head=cfg["transformer"]['n_head'],
    nlayers=cfg["transformer"]['nlayers'])
backbone = nn.Sequential(*list(transformer.children())[-2])
model_sim = selfsupervised.model.SimSiam_LM(backbone,
                                            num_ftrs=num_ftrs,
                                            proj_hidden_dim=proj_hidden_dim,
                                            pred_hidden_dim=pred_hidden_dim,
                                            out_dim=out_dim,
                                            lr=lr_pre)

tb_logger = pl_loggers.TensorBoardLogger(save_dir="../../logs/")
trainer = pl.Trainer(gpus=_gpu,
                     deterministic=True,
                     max_epochs=300,
                     logger=tb_logger)
features = [
    'B3_mean', 'B4_mean', 'B5_mean', 'B6_mean', 'B7_mean', 'B8_mean',
    'B8A_mean'
]
train2 = Custom1617(bavaria_data,
                    features,
                    'Year',
                    field_id='id',
                    time_steps=11,
                    size=5000)
train_dl2 = DataLoader(train2,
                       batch_size=batch_size_pre,
                       shuffle=True,
                       drop_last=False,
                       num_workers=2)
# fit the first time with one augmentation
trainer.fit(model_sim, train_dl2)

# %%
tb_logger = pl_loggers.TensorBoardLogger(save_dir="../../logs/")
trainer = pl.Trainer(gpus=_gpu,
                     deterministic=True,
                     max_epochs=300,
                     logger=tb_logger)
features = [
    'B3_mean', 'B4_mean', 'B5_mean', 'B6_mean', 'B7_mean', 'B8_mean',
    'B8A_mean'
]
train3 = Custom1718(bavaria_data,
                    features,
                    'Year',
                    field_id='id',
                    time_steps=11,
                    size=5000)
train_dl3 = DataLoader(train3,
                       batch_size=batch_size_pre,
                       shuffle=True,
                       drop_last=False,
                       num_workers=2)
# fit the first time with one augmentation
# trainer.fit(model_sim, train_dl3)

path_to_modeldir = '../../models/test300_29.9.ckpt'
torch.save(backbone, path_to_modeldir)

# %%
# Transfer
transformer = selfsupervised.model.Attention(
    input_dim=cfg["transformer"]['input_dim'],
    num_classes=cfg["transformer"]['num_classes'],
    n_head=cfg["transformer"]['n_head'],
    nlayers=cfg["transformer"]['nlayers'])
path_to_modeldir = '../../models/test300_29.9.ckpt'
backbone = torch.load(path_to_modeldir)
# backbone = nn.Sequential(*list(backbone.children())[-2])

tb_logger3 = pl_loggers.TensorBoardLogger(save_dir="../../logs/")
trainer = pl.Trainer(deterministic=True, max_epochs=100, logger=tb_logger3)
model4 = selfsupervised.model.Attention_Transfer(
    lr=lr_pre,
    input_dim=7,
    num_classes=3,
    d_model=64,
    backbone=backbone,
    head=nn.Sequential(*list(transformer.children())[-1]),
    batch_size=batch_size_fine,
    finetune=False)
trainer.fit(model4, train_dl)

# %%
backbone

# %%
