# %%
# compare the crop type classification of RF and SimSiam
import sys
import os
sys.path.append('./model')
sys.path.append('..')

from model import *
from processing import *
import math

import torch.nn as nn
import torchvision
import lightly

import contextily as ctx
import matplotlib.pyplot as plt
import breizhcrops
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
from breizhcrops import BreizhCrops
from breizhcrops.datasets.breizhcrops import BANDS as allbands

import torch
import tqdm

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

import sklearn.datasets
import pandas as pd
import numpy as np
import umap
import umap.plot

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pytorch_lightning import Trainer, seed_everything
import copy
#tsai could be helpful
#from tsai.all import *
#computer_setup()

#some definitions for Transformers
batch_size = 1349
test_size = 0.25
SEED = 42
num_workers=4
shuffle_dataset =True
_epochs = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr =  0.0016612

#definitions for simsiam
num_ftrs = 64
proj_hidden_dim =14
pred_hidden_dim =14
out_dim =14
# scale the learning rate
#lr = 0.05 * batch_size / 256

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

#seed_everything(42, workers=True)

#test function
def test_epoch(model, criterion, test_dl, device):
    model.eval()
    with torch.no_grad():
        losses = list()
        y_true_list = list()
        y_pred_list = list()
        y_score_list = list()

        with tqdm.tqdm(enumerate(test_dl), total=len(test_dl), leave=True) as iterator:
            for idx, batch in iterator:
                x, y_true = batch
                logprobabilities = model.forward(x.to(device))
                loss = criterion(logprobabilities, y_true.to(device))
                iterator.set_description(f"test loss={loss:.2f}")
                losses.append(loss)
                y_true_list.append(y_true)
                #print(logprobabilities.shape)
                y_pred_list.append(logprobabilities.argmax(-1))
                y_score_list.append(logprobabilities.exp())

        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list), torch.cat(y_score_list)

# %%
#load data for bavaria
dm_bavaria = BavariaDataModule(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size, num_workers = num_workers)
dm_bavaria2 = Bavaria1617DataModule(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size, num_workers = num_workers)
dm_bavaria3 = Bavaria1percentDataModule(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size, num_workers = num_workers)
# %%
# Vorgehen:
# 1. Pre-Train transformer unsupervised mit allen Daten (typische Augmentation + physikalisch)
# 2. Finetune with data 16/17 + 1 prozent 18

# %%
transformer = Attention(num_classes = 6, n_head=4, nlayers=3)
backbone = nn.Sequential(*list(transformer.children())[-2])

dm_augmented = DataModule_augmentation(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size, num_workers = num_workers)
dm_augmented.setup('fit')
model_sim = SimSiam_LM(backbone,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)
trainer = pl.Trainer(gpus=1 if str(device).startswith("cuda") else 0, deterministic=True, max_epochs = _epochs)
trainer.fit(model_sim, datamodule=dm_augmented)
#trainer.save_checkpoint("../model/pretrained/simsiam.ckpt")
torch.save(backbone, "../model/pretrained/backbone.ckpt")

#copy pretrained backbone for experiments
backbone_copy1 = copy.deepcopy(backbone)
backbone_copy2 = copy.deepcopy(backbone)
backbone_copy3 = copy.deepcopy(backbone)

# %%
#use pretrained backbone and finetune 
transformer1 = Attention(num_classes = 6, n_head=4, nlayers=3)
head = nn.Sequential(*list(transformer1.children())[-1])

transfer_model = Attention_Transfer(num_classes = 6, d_model=num_ftrs, backbone = backbone_copy1, head=head, batch_size = batch_size, finetune=True, lr=lr)
trainer = pl.Trainer( gpus=1 if str(device).startswith("cuda") else 0, deterministic=True, max_epochs= _epochs)

trainer.fit(transfer_model, datamodule = dm_bavaria)
trainer.test(transfer_model, datamodule = dm_bavaria)
# %%
transformer2 = Attention(num_classes = 6, n_head=4, nlayers=3)
head2 = nn.Sequential(*list(transformer2.children())[-1])

#use pretrained backbone and finetune 
transfer_model2 = Attention_Transfer(num_classes = 6, d_model=num_ftrs, backbone = backbone_copy2, head=head2, batch_size = batch_size, finetune=True, lr=lr)
trainer = pl.Trainer( gpus=1 if str(device).startswith("cuda") else 0, deterministic=True, max_epochs= _epochs)

trainer.fit(transfer_model2, datamodule = dm_bavaria2)
trainer.test(transfer_model2, datamodule = dm_bavaria2)
# %%
transformer3 = Attention(num_classes = 6, n_head=4, nlayers=3)
head3 = nn.Sequential(*list(transformer3.children())[-1])

transfer_model3 = Attention_Transfer(num_classes = 6, d_model=num_ftrs, backbone = backbone_copy3, head=head3, batch_size = batch_size, finetune=True, lr=lr)
trainer = pl.Trainer( gpus=1 if str(device).startswith("cuda") else 0, deterministic=True, max_epochs= _epochs)

trainer.fit(transfer_model3, datamodule = dm_bavaria3)
trainer.test(transfer_model3, datamodule = dm_bavaria3)
# %%
