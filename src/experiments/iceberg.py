# %%
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
import torch.nn.functional as F
import argparse
from pathlib import Path

utils.seed_torch()


################################################################
## Configuration 
################################################################


if os.path.isfile(sys.argv[1]) and os.access(sys.argv[1], os.R_OK):
    # Open both config files as dicts and combine them into a single dict.
    with open(sys.argv[1],'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
else:
    with open("../../config/iceberg/param_config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)



batch_size_pre = cfg["pretraining"]['batch_size']
batch_size_fine = cfg["finetuning"]['batch_size']
input_size = cfg["pretraining"]['input_size']

num_ftrs = cfg["pretraining"]['num_ftrs']
proj_hidden_dim = cfg["pretraining"]['proj_hidden_dim']
pred_hidden_dim = cfg["pretraining"]['pred_hidden_dim']
out_dim= cfg["pretraining"]['out_dim']
log_interval = cfg["pretraining"]['log_interval']
log_interval_fine = cfg["finetuning"]['log_interval']
channels = cfg["UNet"]['channels']
dropout = cfg["UNet"]['dropout']
num_workers = 0

#Path to all log files, directories
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
path_to_datadir = str(Path(ROOT_DIR).parents[0] ) + '/data/iceberg/'
path_to_modeldir = str(Path(ROOT_DIR).parents[0] ) + '/models/iceberg/iceberg_pretrained.ckpt' 
path_to_logdir = str(Path(ROOT_DIR).parents[0] ) + '/logs/iceberg/'

# %%
# %%
################################################################
## Load data & Augmentation 
## Augmentation based on 
# https://docs.lightly.ai/tutorials/package/tutorial_simsiam_esa.html
################################################################

# define the augmentations for self-supervised learning
collate_fn = lightly.data.ImageCollateFunction(
    input_size=256,
    # require invariance to flips and rotations
    hf_prob=0.5,
    vf_prob=0.5,
    rr_prob=0.5,
    # slight random cropping
    min_scale=0.5,
    # use a weak color jitter for invariance w.r.t small color changes
    cj_prob=0.2,
    cj_bright=0.1,
    cj_contrast=0.1,
    cj_hue=0.1,
    cj_sat=0.1,
    #gaussian blur to be invariant for texture details
    gaussian_blur=0.3

)


test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
])


#######################
# load data 
#######################

import re
x_train = []
y_train = []
train_mask=[]

for filename in sorted(glob.glob(path_to_datadir + 'PNG_train/S1*.png')):
   
   # split filename to find the matching files
   name=re.search('/S1_(.*).png', filename).group(1)
   name_part1=re.search('/S1_(.*)99_', filename).group(1)
   name_part2=re.search('_99_(.*).png', filename).group(1)

   # load input image
   im = Image.open(filename)
   x_train.append(np.array(im))
   
   # load corresponding ground truth
   filename2=path_to_datadir + 'GT_train/GT_'+ name + '.png'
   gt = Image.open(filename2)
   y_train.append(np.array(gt))
  
   # load corresponding mask (no satellite coverage)
   filename3=path_to_datadir + 'Mask_train/NaN_mask_'+ name_part1 + name_part2 + '.png'
   mask = Image.open(filename3)
   train_mask.append(np.array(mask))

x_train=np.array(x_train)
y_train=np.array(y_train)
train_mask=np.array(train_mask)

print('Training data sizes (should be 161 x 256 x 256 for all of them)')
print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(train_mask))


# we use the LighltyDataset for the pretraining
dataset_train= lightly.data.LightlyDataset(
    input_dir=path_to_datadir + 'PNG_train/',
    #transform=test_transforms
)
dataset_unlabeled= lightly.data.LightlyDataset(
    input_dir=path_to_datadir + 'PNG_unlabeled/',
    #transform=test_transforms
)
train_dev = torch.utils.data.ConcatDataset([dataset_train, dataset_unlabeled])


# data loader
dataloader_train_unsupervised = torch.utils.data.DataLoader(
    train_dev,
    batch_size=batch_size_pre,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)


# create dataloader with x and y for finetuning
tensor_x = torch.Tensor(np.repeat(x_train[:,np.newaxis, :, :], 3, axis=1)) # transform to torch tensor
tensor_y = torch.Tensor(y_train[:,np.newaxis, :, :])

my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
my_dataloader = torch.utils.data.DataLoader(my_dataset, 
    batch_size=batch_size_fine,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers) 
# %%
tensor_x.shape

# %%
################################################################
## Plot some examples
################################################################
fig = plt.figure(figsize=(8, 8))
columns = 8
rows = 8

#for i in range(1, columns*rows +1):
#    img = formated_img[i]
#    fig.add_subplot(rows, columns, i)
#    plt.imshow(img)
#plt.show()

glob_to_data = path_to_datadir + 'PNG_unlabeled/S1*.png'
fnames = glob.glob(glob_to_data)
input_images = [Image.open(fname) for fname in fnames[:2]]
# plot the images
fig = lightly.utils.debug.plot_augmented_images(input_images, collate_fn)


# %%
iterator=iter(my_dataloader)
inputs= next(iterator)
inputs[0].shape

# %%

# %%

###################################################################
# Model SimSiam --- Pretraining
# Use a UNET encoder and pretrain it using the defined augmentations
###################################################################


filters=[32, 64, 128, 256]
tb_logger = pl_loggers.TensorBoardLogger(save_dir=path_to_logdir)

_encoder = ssl.model.ResUnetEncoder(channel=channels, filters =filters, dropout = dropout)
model = ssl.model.SimSiam_UNet(_encoder, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim)
trainer = pl.Trainer(gpus=cfg["pretraining"]['gpus'], deterministic=True, max_epochs = cfg["pretraining"]['epochs'], logger=tb_logger, log_every_n_steps=log_interval)
#fit the first time with one augmentation
trainer.fit(model, dataloader_train_unsupervised )
# %%
#save pretrained model
torch.save(_encoder, path_to_modeldir)
# %%
###################################################################
# Model SimSiam --- FineTune Unet
# Use pretrained encoder and finetune with labels
###################################################################


backbone_pretrain = torch.load(path_to_modeldir)

filters=[32, 64, 128, 256]
#include the pretrained encoder!
model_finetune = ssl.model.UNet_Transfer(backbone=backbone_pretrain)
trainer = pl.Trainer(gpus=cfg["pretraining"]['gpus'], deterministic=True, max_epochs = cfg["pretraining"]['epochs'] , logger=tb_logger, log_every_n_steps=log_interval_fine)
#fit the first time with one augmentation
trainer.fit(model_finetune, my_dataloader)

# %%

# %%
