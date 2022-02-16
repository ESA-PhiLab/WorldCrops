# %%
# compare the crop type classification of RF and SimSiam
import sys

#sys.path.append('/home/daniel/QM-Encoder/worldcrops/WorldCrops/WorldCrops/py_files')
sys.path.append('/workspace/WorldCrops/py_files')
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
    "../../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx")
bavaria_test = pd.read_excel(
    "../../../data/cropdata/Bavaria/sentinel-2/Test_bavaria.xlsx")

bavaria_reordered = pd.read_excel(
    '../../../data/cropdata/Bavaria/sentinel-2/data2016-2018.xlsx', index_col=0)
bavaria_test_reordered = pd.read_excel(
    '../../../data/cropdata/Bavaria/sentinel-2/TestData.xlsx', index_col=0)

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
#%%
feature_list = train.columns[train.columns.str.contains('B')]
# len(train.id.unique())
#%%
print(train.shape)
trainx = train[feature_list].values
df = trainx.reshape(1799,14, len(feature_list))
#%%

class AugmentationSampling():
    '''Obtain mean and std for each timestep from dataset and draw augmentation from that.
       REQUIRES: data[type,channel,timestep,samples]
    '''
    def __init__(self, data) -> None:
        self.types = data.shape[0]
        self.channels = data.shape[1]
        self.time_steps = data.shape[2]
        self.mu = torch.zeros((self.types, self.channels, self.time_steps))
        self.std = torch.zeros((self.types, self.channels, self.time_steps))
        for f in range(self.types):
            for c in range(self.channels):
                for t in range(self.time_steps):
                    self.mu[f,c,t] = torch.mean(torch.tensor(data[f,c,t,:]))
                    self.std[f,c,t] = torch.std(torch.tensor(data[f,c,t,:]))

    def create_augmentation(self, type, n_samples):
        samples = torch.zeros((n_samples, self.channels,self.time_steps))
        for n in range(n_samples):
            for c in range(self.channels):
                for t in range(self.time_steps):
                    samples[n,c,t] = torch.normal(mean=self.mu[type,c,t], std=self.std[type,c,t])
        return samples


class TSDataSet(Dataset):
    '''
    :param data: dataset of type pandas.DataFrame
    :param target_col: targeted column name
    :param field_id: name of column with field ids
    :param feature_list: list with target features
    :param callback: preprocessing of dataframe
    '''
    def __init__(self, data, factor=1, feature_list = [], target_col = 'NC', field_id = 'id', time_steps = 14, callback = None):
        self.df = data
        self.factor = factor
        self.df = self.reproduce(data, self.factor)
        self.target_col = target_col
        self.feature_list = feature_list
        self.time_steps = time_steps
        

        if callback != None:
            self.df = callback(self.df)

        self._fields_amount = len(self.df[field_id].unique())*self.factor

        #get numpy
        self.y = self.df[self.target_col].values
        self.field_ids = self.df[field_id].values
        self.df = self.df[self.feature_list].values

        if self.factor < 1:
            print('Factor needs to be at least 1')
            return
        if self.y.size == 0:
            print('Target column not in dataframe')
            return
        if self.field_ids.size == 0:
            print('Field id not defined')
            return
        
        #reshape to 3D
        #field x T x D
        self.df = self.df.reshape(int(self._fields_amount),self.time_steps, len(self.feature_list))
        self.y = self.y.reshape(int(self._fields_amount),1, self.time_steps)
        self.field_ids = self.field_ids.reshape(int(self._fields_amount),1, self.time_steps)

        # ::: Statistics for augmentation sampling
        temp_data = np.array(data)
        n_features = len(data.NC.unique())
        n_channels = 13
        n_tsteps = 14
        n_samples = 300
        entries = data.shape[0]
        # : Required data format
        data_sorted = np.zeros((n_features,n_channels,n_tsteps,n_samples))
        for m in range(n_features):
            cnt = 0
            fcnt = 0
            for n in range(entries):
                if(temp_data[n,3]==m):
                    if (cnt==14):
                        fcnt += 1
                        cnt = 0
                    data_sorted[m,:n_channels,cnt,fcnt] = temp_data[n,4:17] 
                    cnt += 1

        # :: Initialize statistical augmentation object
        self.aug_sample = AugmentationSampling(data_sorted)
        # : Usage: self.aug_sample.create_augmentation([type], [n_samples])
        # : Example creates 2 samples for crop type 0
        # aug_samples = self.aug_sample.create_augmentation(0,2)

        # import matplotlib.pyplot as plot
        # fig, ax = plt.subplots(figsize=(8,5))
        # for n in range(6):
        #     ax.plot(ac.mu[n,0,:])
        # fig, ax = plt.subplots(figsize=(8,5))
        # for n in range(6):
        #     ax.plot(ac.std[n,0,:])

    def reproduce(self, df, _size):
        ''' reproduce the orginal df with factor X times'''
        newdf = pd.DataFrame()
        for idx in range(_size):
            newdf = pd.concat([newdf, df.copy()], axis=0)
            #print(len(newdf),_size)
        return newdf

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        x = self.df[idx,:,:]
        y = self.y[idx,0,0]
        field_id = self.field_ids[idx,0,0]

        aug_samples = self.aug_sample.create_augmentation(y.item(),2)
        aug_x1 = aug_samples[0].permute(1,0)
        aug_x2 = aug_samples[1].permute(1,0)

        torchx = self.x2torch(x)
        torchy = self.y2torch(y)
        return aug_x1, aug_x2, torchx, torchy #, torch.tensor(field_id, dtype=torch.long)
        
    def x2torch(self, x):
        '''
        return torch for x
        '''
        #nb_obs, nb_features = self.x.shape
        return torch.from_numpy(x).type(torch.FloatTensor)

    def y2torch(self, y):
        '''
        return torch for y
        '''
        return torch.tensor(y, dtype=torch.long)

#%%
# print(train)
cc = TSDataSet(train, factor=2, feature_list=feature_list.tolist())

#%%
train
#%%
#plt.plot(cc[0][0].numpy())
plt.plot(cc[49][1].numpy())

#%%
train.columns
#%%
train[train.id==0]['NDVI'].plot()

#%%
# print(df.shape)
print(df[:10,0,0])
# print(df)
# y = y.reshape(int(_fields_amount),1, 14)
# field_ids = field_ids.reshape(int(_fields_amount),1, time_steps)

# %%
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