# %%
# compare the crop type classification of RF and SimSiam
import sys
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

from torch.utils.data.sampler import SubsetRandomSampler

#tsai could be helpful
#from tsai.all import *
#computer_setup()

#some definitions for Transformers
batch_size = 32
test_size = 0.25
SEED = 42
num_workers=4
shuffle_dataset =True
_epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
#load data for bavaria
bavaria_train = pd.read_excel(
    "../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx")
bavaria_test = pd.read_excel(
    "../../data/cropdata/Bavaria/sentinel-2/Test_bavaria.xlsx")

bavaria_reordered = pd.read_excel(
    '../../data/cropdata/Bavaria/sentinel-2/data2016-2018.xlsx', index_col=0)
bavaria_test_reordered = pd.read_excel(
    '../../data/cropdata/Bavaria/sentinel-2/TestData.xlsx', index_col=0)

#delete first two entries with no change
#bavaria_train = bavaria_train.loc[~((bavaria_train.id == 0)|(bavaria_train.id == 1))]
#bavaria_reordered = bavaria_reordered.loc[~((bavaria_reordered.index == 0)|(bavaria_reordered.index == 1))]

#bavaria_train.to_excel (r'test.xlsx', index = False, header=True)
train = utils.clean_bavarian_labels(bavaria_train)
test = utils.clean_bavarian_labels(bavaria_test)
#delete class 6 which is the class Other with various unidentified crops
train = train[train.NC != 6]
test = test[test.NC != 6]

train_RF = utils.clean_bavarian_labels(bavaria_reordered)
test_RF = utils.clean_bavarian_labels(bavaria_test_reordered)
#delete class 6
train_RF = train_RF[train_RF.NC !=6]
test_RF = test_RF[test_RF.NC != 6]

train = utils.rewrite_id_CustomDataSet(train)
test = utils.rewrite_id_CustomDataSet(test)

# %%
feature_list = train.columns[train.columns.str.contains('B')]
id=1
print(train[(train.Year == 2016)&(train.NC==id)][feature_list].mean())
print(train[(train.Year == 2017)&(train.NC==id)][feature_list].mean())
print(train[(train.Year == 2018)&(train.NC==id)][feature_list].mean())
print("######")
print(train[(train.Year == 2016)&(train.NC==id)][feature_list].std())
print(train[(train.Year == 2017)&(train.NC==id)][feature_list].std())
print(train[(train.Year == 2018)&(train.NC==id)][feature_list].std())
print("######")
print(train[(train.Year == 2016)][feature_list].mean())
print(train[(train.Year == 2017)][feature_list].mean())
print(train[(train.Year == 2018)][feature_list].mean())
# %%
train.id.unique()
# %%
data = train[feature_list].to_numpy()
data = data.reshape(len(feature_list), len(train.id.unique()) ,14)

# %%
data[0].shape
# %%
def obtain_distribution(data):
    channels = data.shape[0]
    time_steps = data.shape[2]
    mu = torch.zeros((channels, time_steps))
    std = torch.zeros((channels, time_steps))
    for m in range(channels):
        for n in range(time_steps):
            mu[m,n] = torch.mean(torch.tensor(data[m,:,n]))
            std[m,n] = torch.std(torch.tensor(data[m,:,n]))
    
    return mu, std

mu = torch.zeros((7,13,14))
std = torch.zeros((7,13,14))
for n in range(7):
    mu[n], std[n] = obtain_distribution(data[n])
#%%
### DRAW SAMPLE FROM CROP TYPE SPECIFIC DISTRIBUTIONS
def create_augmentation(type, mu, std):
    sample = torch.zeros((13,14))
    for m in range(16):
        for n in range(14):
            sample[m,n] = torch.normal(mean=mu[type,m,n], std=std[type,m,n])
    return sample
crop_type = 2
sample = create_augmentation(crop_type, mu, std)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2, figsize=(15,7))
channel = 3
ax[0].plot(sample[channel], alpha=1.0, label='Sampled Type 2')
ax[1].plot(mu[crop_type,channel], alpha=1.0, label='Averaged Type 2')
ax[0].legend()
ax[1].legend()