# %%
import os
import math
import matplotlib.pyplot as plt
import matplotlib.offsetbox as osb
from matplotlib import rcParams as rcp
import numpy as np
import tqdm
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as functional
from sklearn import random_projection
import pytorch_lightning as pl

from torchvision.datasets import CIFAR10
from torchvision import transforms

import sys
sys.path.append('/workspace/WorldCrops/py_files')
sys.path.append('..')

#import model
#from model import *
from processing import *
#from tsai.all import *

import torch
import breizhcrops as bc

# %%
class Attention_LM(pl.LightningModule):

    def __init__(self, input_dim = 13, num_classes = 7, d_model = 64, n_head = 2, d_ffn = 128, nlayers = 2, dropout = 0.018, activation="relu", lr = 0.0002):
        super().__init__()
        """
        Args:
            input_dim: amount of input dimensions -> Sentinel2 has 13 bands
            num_classes: amount of target classes
            dropout: default = 0.018
            d_model: default = 64 #number of expected features
            n_head: default = 2 #number of heads in multiheadattention models
            d_ff: default = 128 #dim of feedforward network 
            nlayers: default = 2 #number of encoder layers
            + : https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        Input:
            batch size(N) x T x D
        Output
            batch size(N) x Targets
        """
        self.model_type = 'Transformer_LM'
        self.lr = lr
        self.ce = nn.CrossEntropyLoss()

        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout = dropout, activation=activation, batch_first=True)
        self.inlinear = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU()
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers, nn.LayerNorm(d_model))
        self.outlinear = nn.Linear(d_model, num_classes)

    def forward(self,x):
        # N x T x D -> N x T x d_model / Batch First!
        x = self.inlinear(x) 
        x = self.relu(x)
        x = self.transformer_encoder(x)
        x = x.max(1)[0]
        x = self.relu(x)
        x = self.outlinear(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def training_step(self, batch, batch_idx):
        (x1,x2), x, y = batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        (x1,x2), x, y = val_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# %%
test = Attention_LM(num_classes=7)
# %%
test
# %%
train = pd.read_excel(
    "/workspace/WorldCrops/data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx")

train = utils.clean_bavarian_labels(train)
#delete class 0
train = train[train.NC != 0]

#rewrite the 'id' as we deleted one class
newid = 0
groups = train.groupby('id')
for id, group in groups:
    train.loc[train.id == id, 'id'] = newid
    newid +=1

years = [2016,2017,2018]
train = utils.augment_df(train, years)

#train = utils.clean_bavarian_labels(bavaria_train)
feature_list = train.columns[train.columns.str.contains('B')]
ts_dataset = TimeSeriesPhysical(train, feature_list.tolist(), 'NC')
dataloader_train = torch.utils.data.DataLoader(
    ts_dataset, batch_size=32, shuffle=True,drop_last=False, num_workers=2)

# %%
train.columns[train.columns.str.contains('B')]
# %%
dataiter = iter(dataloader_train)
(x1,x2),x, y = next(dataiter)

# %%

trainer = pl.Trainer(auto_scale_batch_size='power', gpus=0, deterministic=True, max_epochs=5)
trainer.fit(test, dataloader_train)
# %%
test
# %%
