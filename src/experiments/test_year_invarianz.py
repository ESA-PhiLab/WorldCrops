# %%
# compare the crop type classification of RF and SimSiam
import sys
import os
import math

import torch.nn as nn
import torchvision
import lightly

import matplotlib.pyplot as plt
import breizhcrops
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import pandas as pd
import numpy as np
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
import copy
import selfsupervised
from pytorch_lightning import loggers as pl_loggers
from selfsupervised.processing import utils
utils.seed_torch()
import yaml

with open("../../config/croptypes/param_config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


batch_size_pre = cfg["pretraining"]['batch_size']
batch_size_fine = cfg["finetuning"]['batch_size']




# %%
#load data for bavaria
#experiment with train/test split for all data
dm_bavaria = selfsupervised.data.croptypes.BavariaDataModule(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_pre, num_workers = 4, experiment='Experiment1')

# data for invariance between crops
dm_crops1 = selfsupervised.data.croptypes.AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_pre, num_workers = 4, experiment='Experiment3')


# %%
#create artificail sin cos dataset with 3 classes

import numpy as np
seed = 12345512
np.random.seed(seed)

n = 14
x_data = np.linspace(0, 14, num=n)
y_data = np.cos(x_data) 
y_data2 = np.sin(x_data)
y_data3 = np.sin(x_data) +0.3


# %%
plt.plot(y_data)
plt.plot(y_data2)
plt.plot(y_data3)
# %%

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# %%

X_train


# %%
# Vorgehen:
# 1. Pre-Train transformer unsupervised mit allen Daten (typische Augmentation + physikalisch)
# 2. Finetune

transformer = selfsupervised.model.Attention(input_dim= cfg["transformer"]['input_dim'], num_classes = cfg["transformer"]['num_classes'], n_head=cfg["transformer"]['n_head'], nlayers=cfg["transformer"]['nlayers'])
backbone = nn.Sequential(*list(transformer.children())[-2])

#parameter for SimSiam
num_ftrs = cfg["pretraining"]['num_ftrs']
proj_hidden_dim = cfg["pretraining"]['proj_hidden_dim']
pred_hidden_dim = cfg["pretraining"]['pred_hidden_dim']
out_dim = cfg["pretraining"]['out_dim']
lr_pre =  cfg["pretraining"]['learning_rate']
epochs_pre = cfg["pretraining"]['epochs']
_gpu = cfg["pretraining"]['gpus']

model_sim = selfsupervised.model.SimSiam_LM(backbone,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr_pre)

tb_logger = pl_loggers.TensorBoardLogger(save_dir="../../logs/")
trainer = pl.Trainer(gpus=_gpu, deterministic=True, max_epochs = epochs_pre, logger=tb_logger)

#fit the first time with one augmentation
trainer.fit(model_sim, datamodule=dm_crops1)

# %%

backbone_copy1 = copy.deepcopy(backbone)
head = nn.Sequential(*list(transformer.children())[-1])
epochs_fine = cfg["finetuning"]['epochs']
lr_fine =  cfg["finetuning"]['learning_rate']

transfer_model = selfsupervised.model.Attention_Transfer(input_dim=cfg["transformer"]['input_dim'], num_classes = cfg["transformer"]['num_classes'], d_model=num_ftrs, backbone = backbone_copy1, head=head, batch_size = batch_size_fine, finetune=False, lr=lr_fine)
trainer = pl.Trainer(gpus=_gpu, deterministic=True, max_epochs = epochs_fine, logger=tb_logger)

trainer.fit(transfer_model, datamodule = dm_bavaria)
trainer.test(transfer_model, datamodule = dm_bavaria)

# %%
