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
# import umap
# import umap.plot

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pytorch_lightning import Trainer, seed_everything
import copy
#tsai could be helpful
#from tsai.all import *
#computer_setup()

#some definitions for Transformers
batch_size = 250
test_size = 0.25
# SEED = 42
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

# %%
#load data for bavaria
dm_aug = AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = 250, num_workers = num_workers, experiment='Experiment1')
dm16 = Experiment2016(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size, num_workers = num_workers)

# %%
#supervised test
model = Attention_LM(num_classes = 6, n_head=4, nlayers=3, batch_size = batch_size, lr=lr, seed=42)
trainer = pl.Trainer( gpus=1 if str(device).startswith("cuda") else 0, deterministic=True, max_epochs= 200)
trainer.fit(model, datamodule=dm16)
trainer.test(model, datamodule=dm16)

# %%
no_gpus = 4

transformer = Attention(num_classes = 6, n_head=4, nlayers=3)
backbone = nn.Sequential(*list(transformer.children())[-2])
# # %%
model_sim = SimSiam_LM(backbone,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)
#trainer = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs)
trainer = pl.Trainer(gpus=4 if str(device).startswith("cuda") else 0, deterministic=True, max_epochs = 200)
#%%
trainer.fit(model_sim, datamodule=dm_aug)
backbone_copy1 = copy.deepcopy(backbone)
backbone_copy2 = copy.deepcopy(backbone)
torch.save(backbone, "../model/pretrained/backbone_2016.ckpt")

# %%
#use pretrained backbone and finetune 
transformer1 = Attention(num_classes = 6, n_head=4, nlayers=3)
head = nn.Sequential(*list(transformer1.children())[-1])

transfer_model = Attention_Transfer(num_classes = 6, d_model=num_ftrs, backbone = backbone_copy1, head=head, batch_size = batch_size, finetune=True, lr=lr)
trainer = pl.Trainer( gpus=1 if str(device).startswith("cuda") else 0, deterministic=True, max_epochs= 200)

trainer.fit(transfer_model, datamodule = dm16)
trainer.test(transfer_model, datamodule = dm16)

# %%
transformer1 = Attention(num_classes = 6, n_head=4, nlayers=3)
head = nn.Sequential(*list(transformer1.children())[-1])

transfer_model = Attention_Transfer(num_classes = 6, d_model=num_ftrs, backbone = backbone_copy2, head=head, batch_size = batch_size, finetune=False, lr=lr)
trainer = pl.Trainer( gpus=1 if str(device).startswith("cuda") else 0, deterministic=True, max_epochs= 200)

trainer.fit(transfer_model, datamodule = dm16)
trainer.test(transfer_model, datamodule = dm16)

#test 113 - 116 sagt dass sich nichts verändert 0.77