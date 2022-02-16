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
_epochs = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr =  0.0016612
lr_sim = 0.05 * batch_size / 256

#definitions for simsiam
num_ftrs = 64
proj_hidden_dim =64
pred_hidden_dim =16
out_dim =64
# scale the learning rate


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
dm_aug = AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size, num_workers = num_workers, experiment='Experiment2')
dm16 = Experiment2016(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size, num_workers = num_workers)

# %%
#supervised test
model = Attention_LM(num_classes = 6, n_head=4, nlayers=3, batch_size = batch_size, lr=lr, seed=42)
trainer = pl.Trainer( gpus=1 if str(device).startswith("cuda") else 0, deterministic=True, max_epochs= _epochs)
trainer.fit(model, datamodule=dm16)
trainer.test(model, datamodule=dm16)

# %%
model_sim

# %%
transformer = Attention(num_classes = 6, n_head=4, nlayers=3)
backbone = nn.Sequential(*list(transformer.children())[-2])

class SimSiam_LM2(pl.LightningModule):
    def __init__(self, backbone = nn.Module, num_ftrs=64, proj_hidden_dim=14, 
    pred_hidden_dim=14, out_dim=14, lr=0.02, weight_decay=5e-4,momentum=0.9,epochs = 10):
        super().__init__()
        
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.out_dim = out_dim

        self.backbone = backbone
        self.model_type = 'SimSiam_LM'
        self.projection = lightly.models.modules.heads.ProjectionHead([
            (num_ftrs, proj_hidden_dim, nn.BatchNorm1d(proj_hidden_dim), nn.ReLU()),
            (proj_hidden_dim, out_dim, nn.BatchNorm1d(out_dim), None)
        ])
        self.prediction = SimSiamPredictionHead(out_dim,pred_hidden_dim,out_dim)

        self.ce = lightly.loss.NegativeCosineSimilarity()

        self.avg_loss = 0.
        self.avg_output_std = 0.
        

    def forward(self, x0, x1):
        f0 = self.backbone(x0)
        f1 = self.backbone(x1)

        z0 = self.projection(f0)
        z1 = self.projection(f1)

        p0 = self.prediction(z0)
        p1 = self.prediction(z1)

        z0 = z0.detach()
        z1 = z1.detach()
        return (z0, p0),(z1, p1)


    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        (z0, p0),(z1, p1) = self.forward(x0,x1)

        loss = 0.5 * (self.ce(z0, p1) + self.ce(z1, p0))
        self.log('train_loss_ssl', abs(loss))
        #output = p0.detach()
        return {'loss':loss}

    '''def training_step_end(self, batch_parts):
        #print(batch_parts['p0'], type(batch_parts['p0']))
        output = batch_parts['p0']
        loss = batch_parts['loss']
        output = torch.nn.functional.normalize(output, dim=1)
        output_std = torch.std(output, 0)
        output_std = output_std.mean()
        
        # use moving averages to track the loss and standard deviation
        w = 0.9
        self.avg_loss = w * self.avg_loss + (1 - w) * loss
        self.avg_output_std = w * self.avg_output_std + (1 - w) * output_std.item()
        return {'loss':self.avg_loss,'avg_output_std': self.avg_output_std}'''
    
    def training_epoch_end(self, outputs):
        #https://docs.ray.io/en/latest/tune/tutorials/tune-pytorch-lightning.html
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        #avg_output_std
        # the level of collapse is large if the standard deviation of the l2
        # normalized output is much smaller than 1 / sqrt(dim)
        #collapse_level = max(0., 1 - math.sqrt(self.out_dim) * avg_output_std)
        #self.log('Collapse Level', round(collapse_evel,2))



    def validation_step(self, val_batch, batch_idx):
        (x0, x1), _, _ = val_batch
 
        (z0, p0),(z1, p1) = self.forward(x0,x1)
        loss = 0.5 * (self.ce(z0, p1) + self.ce(z1, p0))
        self.log('val_loss', loss)
        return {"loss":loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,
                                momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        return [optimizer], [scheduler]

model_sim = SimSiam_LM2(backbone,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr_sim)
#trainer = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs)
trainer = pl.Trainer(gpus=4 if str(device).startswith("cuda") else 0, deterministic=True, max_epochs = _epochs)
#%%
trainer.fit(model_sim, datamodule=dm_aug)

#%%
backbone_copy1 = copy.deepcopy(backbone)
backbone_copy2 = copy.deepcopy(backbone)
torch.save(backbone, "../model/pretrained/backbone_2.ckpt")

# %%
#use pretrained backbone and finetune 
transformer1 = Attention(num_classes = 6, n_head=4, nlayers=3)
head = nn.Sequential(*list(transformer1.children())[-1])

transfer_model = Attention_Transfer(num_classes = 6, d_model=num_ftrs, backbone = backbone_copy1, head=head, batch_size = batch_size, finetune=True, lr=lr)
trainer = pl.Trainer( gpus=1 if str(device).startswith("cuda") else 0, deterministic=True, max_epochs= _epochs)

trainer.fit(transfer_model, datamodule = dm16)
trainer.test(transfer_model, datamodule = dm16)

# %%
transformer1 = Attention(num_classes = 6, n_head=4, nlayers=3)
head = nn.Sequential(*list(transformer1.children())[-1])

transfer_model = Attention_Transfer(num_classes = 6, d_model=num_ftrs, backbone = backbone_copy2, head=head, batch_size = batch_size, finetune=False, lr=lr)
trainer = pl.Trainer( gpus=1 if str(device).startswith("cuda") else 0, deterministic=True, max_epochs= _epochs)

trainer.fit(transfer_model, datamodule = dm16)
trainer.test(transfer_model, datamodule = dm16)

#test 113 - 116 sagt dass sich nichts verändert 0.77