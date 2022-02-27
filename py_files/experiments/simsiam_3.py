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
import copy
#tsai could be helpful
#from tsai.all import *
#computer_setup()

################################
#IARAI / ESA
IARAI = True
no_gpus = 8
no_gpus = [0,1,2,3,4,5,6]
################################


#some definitions for Transformers
batch_size = 1349
test_size = 0.25
num_workers=4
shuffle_dataset =True
_epochs = 300
input_dim = 13
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr =  0.0016612

#definitions for simsiam
num_ftrs = 64
proj_hidden_dim =14
pred_hidden_dim =14
out_dim =14
batch_size_sim = 256

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
#experiment with train/test split for all data
dm_bavaria = BavariaDataModule(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size, num_workers = num_workers, experiment='Experiment1')
#experiment with 16/17 train and 2018 test
dm_bavaria2 = BavariaDataModule(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size, num_workers = num_workers, experiment='Experiment2')
#experiment with 16/17 + 5% 2018 train and 2018 test
dm_bavaria3 = BavariaDataModule(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size, num_workers = num_workers, experiment='Experiment3')
#experiment with 16/17 + 10% 2018 train and 2018 test
dm_bavaria4 = BavariaDataModule(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size, num_workers = num_workers, experiment='Experiment4')

#SimSiam augmentation data
#augment between 2016 and 2017 for each crop type
dm_augmented = DataModule_augmentation(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_sim, num_workers = num_workers)
#augmentation between years independent of crop type
dm_years = AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment4')

# data for invariance between crops
#dm_crops1 = AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment3')
#dm_crops2 = AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment5')
#dm_crops3 = AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment6')
#dm_crops4 = AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment7')

#daniel datamodule mit statistiken
dm_crops1 = AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment8')
dm_crops2 = AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment9')
dm_crops3 = AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment10')
dm_crops4 = AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment11')

#normale augmentation wie blur, jitter
#dm_crops1 = AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment8')
#dm_crops2 = AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment9')
#dm_crops3 = AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment10')
#dm_crops4 = AugmentationExperiments(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size_sim, num_workers = num_workers, experiment='Experiment11')
# %%
# Vorgehen:
# 1. Pre-Train transformer unsupervised mit allen Daten (typische Augmentation + physikalisch)
# 2. Finetune 

transformer = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
backbone = nn.Sequential(*list(transformer.children())[-2])
model_sim = SimSiam_LM(backbone,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)


if IARAI:
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs)
else:
    trainer = pl.Trainer(deterministic=True, max_epochs = _epochs)

#fit the first time with one augmentation
trainer.fit(model_sim, datamodule=dm_crops1)
torch.save(backbone, "../model/pretrained_b/pretraining1.ckpt")
#%%
#fit 
transformer2 = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
backbone2 = nn.Sequential(*list(transformer2.children())[-2])
model_sim2 = SimSiam_LM(backbone2,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)

if IARAI:
    trainer2 = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs)
else:
    trainer2 = pl.Trainer(deterministic=True, max_epochs = _epochs)
trainer2.fit(model_sim2, datamodule=dm_crops2)
torch.save(backbone2, "../model/pretrained_b/pretraining2.ckpt")

#%%
transformer3 = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
backbone3 = nn.Sequential(*list(transformer3.children())[-2])
model_sim3 = SimSiam_LM(backbone3,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)

#fit for invariance between same crops
if IARAI:
    trainer3 = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs)
else:
    trainer3 = pl.Trainer(deterministic=True, max_epochs = _epochs)

trainer3.fit(model_sim3, datamodule=dm_crops3)
torch.save(backbone3, "../model/pretrained_b/pretraining3.ckpt")

transformer4 = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
backbone4 = nn.Sequential(*list(transformer4.children())[-2])
model_sim4 = SimSiam_LM(backbone4,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,lr=lr)

#fit for invariance between same crops
if IARAI:
    trainer4 = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs)
else:
    trainer4 = pl.Trainer(deterministic=True, max_epochs = _epochs)

trainer4.fit(model_sim4, datamodule=dm_crops4)
torch.save(backbone4, "../model/pretrained_b/pretraining4.ckpt")
#%%

#%%
#backbone = torch.load("../model/pretrained/backbone_3_aug_17.2.ckpt")
#copy pretrained backbone for experiments
backbone_copy1 = copy.deepcopy(backbone)
backbone_copy2 = copy.deepcopy(backbone2)
backbone_copy3 = copy.deepcopy(backbone3)
backbone_copy4 = copy.deepcopy(backbone4)

# %%
#use pretrained backbone and finetune 
transformer1 = Attention(input_dim=input_dim,num_classes = 6, n_head=4, nlayers=3)
head = nn.Sequential(*list(transformer1.children())[-1])

transfer_model = Attention_Transfer(input_dim=input_dim, num_classes = 6, d_model=num_ftrs, backbone = backbone_copy1, head=head, batch_size = batch_size, finetune=True, lr=lr)
if IARAI:
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs)
else:
    trainer = pl.Trainer(deterministic=True, max_epochs= _epochs)
    
trainer.fit(transfer_model, datamodule = dm_bavaria)
trainer.test(transfer_model, datamodule = dm_bavaria)

# %%
transformer2 = Attention(input_dim=input_dim, num_classes = 6, n_head=4, nlayers=3)
head2 = nn.Sequential(*list(transformer2.children())[-1])

#use pretrained backbone and finetune 
transfer_model2 = Attention_Transfer(input_dim=input_dim, num_classes = 6, d_model=num_ftrs, backbone = backbone_copy2, head=head2, batch_size = batch_size, finetune=True, lr=lr)
if IARAI:
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs)
else:
    trainer = pl.Trainer(deterministic=True, max_epochs= _epochs)

trainer.fit(transfer_model2, datamodule = dm_bavaria2)
trainer.test(transfer_model2, datamodule = dm_bavaria2)
# %%
transformer3 = Attention(input_dim=input_dim, num_classes = 6, n_head=4, nlayers=3)
head3 = nn.Sequential(*list(transformer3.children())[-1])

transfer_model3 = Attention_Transfer(input_dim=input_dim, num_classes = 6, d_model=num_ftrs, backbone = backbone_copy3, head=head3, batch_size = batch_size, finetune=True, lr=lr)

if IARAI:
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs)
else:
    trainer = pl.Trainer(deterministic=True, max_epochs= _epochs)

trainer.fit(transfer_model3, datamodule = dm_bavaria3)
trainer.test(transfer_model3, datamodule = dm_bavaria3)

# %%
transformer4 = Attention(input_dim=input_dim, num_classes = 6, n_head=4, nlayers=3)
head4 = nn.Sequential(*list(transformer3.children())[-1])

transfer_model4 = Attention_Transfer(input_dim=input_dim, num_classes = 6, d_model=num_ftrs, backbone = backbone_copy4, head=head4, batch_size = batch_size, finetune=True, lr=lr)

if IARAI:
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp', deterministic=True, max_epochs = _epochs)
else:
    trainer = pl.Trainer(deterministic=True, max_epochs= _epochs)

trainer.fit(transfer_model4, datamodule = dm_bavaria4)
trainer.test(transfer_model4, datamodule = dm_bavaria4)

# %%
