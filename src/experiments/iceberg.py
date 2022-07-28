# %%
# compare the crop type classification of RF and SimSiam
import sys
import os
import math
import glob

import torch.nn as nn
import torchvision
import lightly
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
import selfsupervised as ssl
from pytorch_lightning import loggers as pl_loggers
import yaml
from selfsupervised.processing import utils

import h5py
import PIL
from PIL import ImageOps, Image

utils.seed_torch()

with open("../../config/iceberg/param_config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

batch_size_pre = cfg["pretraining"]['batch_size']
batch_size_fine = cfg["finetuning"]['batch_size']
input_size = cfg["pretraining"]['input_size']

num_ftrs = cfg["pretraining"]['num_ftrs']
proj_hidden_dim = cfg["pretraining"]['proj_hidden_dim']
pred_hidden_dim = cfg["pretraining"]['pred_hidden_dim']
out_dim= cfg["pretraining"]['out_dim']

num_workers = 0


# %%
# read data from h5

x_train = []
formated_img = []
y_train = []
train_mask=[]

S1_fn = []
GT_fn = []

for filename in sorted(glob.glob('../../data/Anne/S1_train_small/S1*.h5')):

  with h5py.File(filename, "r") as f:
    data = np.array(f['/DS1'])
    im=np.transpose(data)
    x_train.append(im)
    S1_fn.append(filename)
    filename = filename[16:]
    #save to temp directory
    formatted = (im * 255 / np.max(im)).astype('uint8')
    data = Image.fromarray(formatted)
    formated_img.append(data)
    data.save('../../data/Anne/tmp/' + filename.split("/", 1)[1] + '.png')

for filename in sorted(glob.glob('../../data/Anne/S1_val_small/S1*.h5')):

  with h5py.File(filename, "r") as f:
    data = np.array(f['/DS1'])
    im=np.transpose(data)
    x_train.append(im)
    S1_fn.append(filename)
    filename = filename[16:]
    #save to temp directory
    formatted = (im * 255 / np.max(im)).astype('uint8')
    data = Image.fromarray(formatted)
    formated_img.append(data)
    data.save('../../data/Anne/tmp2/' + filename.split("/", 1)[1] + '.png')


# %%
#show some examples
fig = plt.figure(figsize=(8, 8))
columns = 8
rows = 8

for i in range(1, columns*rows +1):
    img = formated_img[i]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()



# %%

################################################################
# Augmentations + custom dataset from lightly
# https://docs.lightly.ai/tutorials/package/tutorial_simsiam_esa.html
###################################################################

# define the augmentations for self-supervised learning
collate_fn = lightly.data.ImageCollateFunction(
    input_size=200,
    # require invariance to flips and rotations
    hf_prob=0.5,
    vf_prob=0.5,
    rr_prob=0.5,
    # satellite images are all taken from the same height
    # so we use only slight random cropping
    min_scale=0.5,
    # use a weak color jitter for invariance w.r.t small color changes
    cj_prob=0.2,
    cj_bright=0.1,
    cj_contrast=0.1,
    cj_hue=0.1,
    cj_sat=0.1,

)

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
])


# dataset for training
dataset_train= lightly.data.LightlyDataset(
    input_dir='../../data/Anne/tmp',
    #transform=test_transforms
)

# lightly dataset for embedding
dataset_test = lightly.data.LightlyDataset(
    input_dir='../../data/Anne/tmp2',
    transform=test_transforms
)

# create a dataloader for training
dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=10,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=False,
    num_workers=num_workers
)

# create a dataloader for embedding
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size_pre,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)
# %%

dataloader_train
iterator=iter(dataloader_train)
inputs= next(iterator)
# %%
inputs[0][0].shape
# %%

################################################################
# Model SimSiam 
###################################################################
# pretrained resnet
resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = ssl.model.SimSiam_Images(backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim)

channels = cfg["UNet"]['channels']
dropout = cfg["UNet"]['dropout']

filters=[32, 64, 128, 256]
_encoder = ssl.model.ResUnetEncoder(channel=3, filters =filters, dropout = dropout)
model = ssl.model.SimSiam_UNet_Encoder(_encoder, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim)
# %%


# %%
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

tb_logger = pl_loggers.TensorBoardLogger(save_dir="../../logs/")
trainer = pl.Trainer(gpus=cfg["pretraining"]['gpus'], deterministic=True, max_epochs = cfg["pretraining"]['epochs'], logger=tb_logger)

#fit the first time with one augmentation
trainer.fit(model, dataloader_train)
# %%
#########################################
#use pretrained encoder and finetune with labels
#########################################

fullmodel = ssl.model.UNet_Transfer(backbone=_encoder, filters = filters, dropout = dropout)
trainer = pl.Trainer(gpus=cfg["pretraining"]['gpus'], deterministic=True, max_epochs = cfg["finetuning"]['epochs'], logger=tb_logger)
#fit the first time with one augmentation
trainer.fit(fullmodel, dataloader_train)
# %%
