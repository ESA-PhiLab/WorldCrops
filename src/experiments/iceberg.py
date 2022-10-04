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
## Helper functions
################################################################
import tqdm 
import cv2 as cv 
import copy
def zero_div(x, y):
    return x / y if y else 0

def evaluate_performance(y_true, y_pred):

        pred_areas=[]
        true_areas=[]
        for i in range(len(y_true)):
            prediction=y_pred[i, :,:]
            berg_samples=prediction[y_true[i]==1]
            background_samples=prediction[y_true[i]==0]
            TP= sum(berg_samples)
            FP= sum(background_samples)
            FN= len(berg_samples)-sum(berg_samples)
            TN= len(background_samples)-sum(background_samples)
            
            pred_areas.append(sum(y_pred[i, :,:].flatten()))
            true_areas.append(sum(y_true[i].flatten()))

        true_areas=np.array(true_areas)
        pred_areas=np.array(pred_areas)

        flat_pred=y_pred.flatten()
        val_arr=np.concatenate(y_true, axis=0 )
        flat_true=val_arr.flatten()

        berg_samples=flat_pred[flat_true==1]
        background_samples=flat_pred[flat_true==0]

        TP= sum(berg_samples)
        FP= sum(background_samples)
        FN= len(berg_samples)-sum(berg_samples)
        TN= len(background_samples)-sum(background_samples)
        
        # dice
        dice=zero_div(2*TP,(2*TP+FP+FN))

        print('overall accuracy')
        print(zero_div((TP+TN),(TP+TN+FP+FN)*100))
        print('false pos rate')
        print(zero_div(FP,(TN+FP)*100))
        print('false neg rate')
        print(zero_div(FN ,(TP+FN)*100) )
        print('area deviations')
        print((pred_areas-true_areas)/true_areas*100)
        print('abs mean error in area')
        print(np.mean(abs((pred_areas-true_areas)/true_areas))*100)
        print('area bias')
        print(np.mean((pred_areas-true_areas)/true_areas)*100)
        print('f1')
        print(dice)
        
def test_function( model , data, mask):
    model.eval()

    with torch.no_grad():
        losses = list()
        y_true_list = list()
        y_pred_list = list()
        
        fig=plt.figure(figsize=(24, 16))
        plt.gray()
        z=1;

        with tqdm.tqdm(enumerate(my_test_dataloader), total=len(my_test_dataloader), leave=True) as iterator:
            for idx, batch in iterator:
                x, y_true = batch
                pred = model.forward(x)
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(pred, y_true)
                iterator.set_description(f"test loss={loss:.2f}")
                losses.append(loss)
                y_true_list.append(y_true)
                y_pred_list.append(pred)
                
                ax1 = fig.add_subplot(4,6,z)  
                plt.imshow(np.squeeze(pred))
                plt.xticks([0, 256], " ")
                plt.yticks([0, 256], " ")
                z=z+1

        y_true_list = torch.cat(y_true_list).squeeze().numpy()
        y_pred_list = torch.cat(y_pred_list).squeeze().numpy()

        # masking
        for i in range(len(y_pred_list)):
            pred=y_pred_list[i]
            pred[mask[i]==1]=0
            y_pred_list[i]=pred
        
        
        # threshold and largest connected component
        thres = 65
        connectivity = 4 

        y_pred=copy.deepcopy(np.squeeze(y_pred_list))
        for i in range(len(y_pred_list)):
            src=y_pred_list[i]

            image1copy = np.uint8(src*255)

            ret, thresh = cv.threshold(image1copy,thres,255,cv.THRESH_BINARY)

            (numLabels, labels, stats, centroids) = cv.connectedComponentsWithStats(thresh, connectivity, cv.CV_32S)
            max_label, max_size = max([(i, stats[i, cv.CC_STAT_AREA]) for i in range(1, numLabels)], key=lambda x: x[1])

            y_pred[i] = (labels == max_label).astype("uint8")
  

    return y_true_list, y_pred

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

x_val = []
y_val = []
val_mask=[]

for filename in sorted(glob.glob(path_to_datadir + 'PNG_val/S1*.png')):
   
   # split filename to find the matching files
   name=re.search('/S1_(.*).png', filename).group(1)
   name_part1=re.search('/S1_(.*)99_', filename).group(1)
   name_part2=re.search('_99_(.*).png', filename).group(1)

   # load input image
   im = Image.open(filename)
   x_val.append(np.array(im))
   
   # load corresponding ground truth
   filename2= path_to_datadir + 'GT_val/GT_'+ name + '.png'
   gt = Image.open(filename2)
   y_val.append(np.array(gt))
  
   # load corresponding mask (no satellite coverage)
   filename3= path_to_datadir + 'Mask_val/NaN_mask_'+ name_part1 + name_part2 + '.png'
   mask = Image.open(filename3)
   val_mask.append(np.array(mask))

x_val=np.array(x_val)
y_val=np.array(y_val)
val_mask=np.array(val_mask)

print('Validation data sizes (should be 26 x 256 x 256 for all of them)')
print(np.shape(x_val))
print(np.shape(y_val))
print(np.shape(val_mask))

x_test = []
y_test = []
test_mask=[]

for filename in sorted(glob.glob(path_to_datadir + 'PNG_test/S1*.png')):
   
   # split filename to find the matching files
   name=re.search('/S1_(.*).png', filename).group(1)
   name_part1=re.search('/S1_(.*)99_', filename).group(1)
   name_part2=re.search('_99_(.*).png', filename).group(1)

   # load input image
   im = Image.open(filename)
   x_test.append(np.array(im))
   
   # load corresponding ground truth
   filename2=path_to_datadir + 'GT_test/GT_'+ name + '.png'
   gt = Image.open(filename2)
   y_test.append(np.array(gt))
  
   # load corresponding mask (no satellite coverage)
   filename3=path_to_datadir + 'Mask_test/NaN_mask_'+ name_part1 + name_part2 + '.png'
   mask = Image.open(filename3)
   test_mask.append(np.array(mask))


x_test=np.array(x_test)
y_test=np.array(y_test)
test_mask=np.array(test_mask)

print('Test data sizes (should be 24 x 256 x 256 for all of them)')
print(np.shape(x_test))
print(np.shape(y_test))
print(np.shape(test_mask))

x_unlabeled = []
unlabeled_mask=[]

for filename in sorted(glob.glob(path_to_datadir + 'PNG_unlabeled/S1*.png')):
   
   # split filename to find the matching files
   name_part1=re.search('/S1_(.*)99_', filename).group(1)
   name_part2=re.search('_99_(.*).png', filename).group(1)

   # load input image
   im = Image.open(filename)
   x_unlabeled.append(np.array(im))
  
   # load corresponding mask (no satellite coverage)
   filename3=path_to_datadir + 'Mask_unlabeled/NaN_mask_'+ name_part1 + name_part2 + '.png'
   mask = Image.open(filename3)
   unlabeled_mask.append(np.array(mask))

x_unlabeled=np.array(x_unlabeled)
unlabeled_mask=np.array(unlabeled_mask)

x_train_unsup=np.concatenate((x_train, x_unlabeled), axis=0)
train_unsup_mask=np.concatenate((train_mask, unlabeled_mask), axis=0)

print('Unsupervised training data sizes (Labeled training data + unlabeled data; should be 275 x 256 x 256 for all of them)')
print(np.shape(x_train_unsup))
print(np.shape(train_unsup_mask))


# normalise data to 0-1
x_train=x_train.astype(np.double)
x_train=x_train/255
y_train=y_train.astype(np.double)

x_val=x_val.astype(np.double)
x_val=x_val/255
y_val=y_val.astype(np.double)

x_test=x_test.astype(np.double)
x_test=x_test/255
y_test=y_test.astype(np.double)

x_train_unsup=x_train_unsup.astype(np.double)
x_train_unsup=x_train_unsup/255


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
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)


# create dataloader with x and y for finetuning
#train
tensor_x = torch.Tensor(np.repeat(x_train[:,np.newaxis, :, :], channels, axis=1)) # transform to torch tensor
tensor_y = torch.Tensor(y_train[:,np.newaxis, :, :])

my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
my_dataloader = torch.utils.data.DataLoader(my_dataset, 
    batch_size=batch_size_fine,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers) 

#validate
val_x = torch.Tensor(np.repeat(x_val[:,np.newaxis, :, :], channels, axis=1)) # transform to torch tensor
val_y = torch.Tensor(y_val[:,np.newaxis, :, :])

my_val_dataset = torch.utils.data.TensorDataset(val_x,val_y)
my_val_dataloader = torch.utils.data.DataLoader(my_val_dataset, 
    batch_size=batch_size_fine,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers) 

#test
test_x = torch.Tensor(np.repeat(x_test[:,np.newaxis, :, :], channels, axis=1)) # transform to torch tensor
test_y = torch.Tensor(y_test[:,np.newaxis, :, :])

my_test_dataset = torch.utils.data.TensorDataset(test_x,test_y)
my_test_dataloader = torch.utils.data.DataLoader(my_test_dataset, 
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

###################################################################
# Model UNet without augmentation as baseline
###################################################################

from torch.optim.lr_scheduler import ReduceLROnPlateau

filters=[32, 64, 128, 256]
tb_logger = pl_loggers.TensorBoardLogger(save_dir=path_to_logdir)

_encoder = ResUnetEncoder(channel=channels, filters =filters, dropout = dropout)
model = UNet_Transfer(lr = learning_rate, backbone=_encoder,  dropout = dropout, filters =filters, batch_size  = batch_size_pre, finetune= False)

trainer = pl.Trainer(gpus=cfg["pretraining"]['gpus'], deterministic=True, max_epochs = cfg["pretraining"]['epochs'], logger=tb_logger, log_every_n_steps=log_interval)

# this is how you could use reducting learning rates, but val_loss is empty

#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5)
#trainer = pl.Trainer(gpus=cfg["pretraining"]['gpus'], deterministic=True, max_epochs = 1, logger=tb_logger, log_every_n_steps=log_interval)

#for epoch in range(10):
 # trainer.fit(model, my_dataloader) # epochs should be set to one above!
  #val_loss = trainer.validate(dataloaders=my_val_dataloader)
  #print(val_loss['val_loss'])
  #scheduler.step(val_loss)

trainer.fit(model, my_dataloader, my_val_dataloader)

#save model
torch.save(model, path_to_modeldir + 'iceberg_baseline.ckpt')


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
tb_logger_fine = pl_loggers.TensorBoardLogger(save_dir=path_to_logdir)

filters=[32, 64, 128, 256]
#include the pretrained encoder!
model_finetune = ssl.model.UNet_Transfer(backbone=backbone_pretrain)
trainer = pl.Trainer(gpus=cfg["finetuning"]['gpus'], deterministic=True, max_epochs = cfg["finetuning"]['epochs'] , logger=tb_logger_fine, log_every_n_steps=log_interval_fine)
#fit the first time with one augmentation
trainer.fit(model_finetune, my_dataloader)

# %%
y_true_list, y_pred = test_function(model , my_test_dataloader, test_mask)

# %%
evaluate_performance(y_true_list, y_pred)

# plot results
idxs=range(24)
fig=plt.figure(figsize=(24, 16))
plt.gray()
z=1;
for i in idxs:
  ax1 = fig.add_subplot(4,6,z)  
  plt.imshow(np.squeeze(y_pred[i, :, :]))
  plt.xticks([0, 256], " ")
  plt.yticks([0, 256], " ")
  z=z+1


# %%


# %%

# %%
