# %%
# compare the crop type classification of RF and SimSiam
import sys

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
import umap
import umap.plot

from torch.utils.data.sampler import SubsetRandomSampler

#some definitions for Transformers
batch_size = 32
test_size = 0.25
SEED = 42
num_workers=4
shuffle_dataset =True
_epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%

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

train = rewrite_id_CustomDataSet(train)
test = rewrite_id_CustomDataSet(test)

# %%
feature_list = train.columns[train.columns.str.contains('B')]
# %%
feature_list.tolist()

# %%
import seaborn as sns
from sklearn import preprocessing

_2016 = train[train.Year == 2016][feature_list.tolist()]
_2017 = train[train.Year == 2017][feature_list.tolist()]
_2018 = train[train.Year == 2018][feature_list.tolist()]

x = _2016.values #returns a numpy array
x2 = _2017.values
x3 = _2018.values

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x_scaled2 = min_max_scaler.fit_transform(x2)
x_scaled3 = min_max_scaler.fit_transform(x3)

df = pd.DataFrame(x_scaled)
df2 = pd.DataFrame(x_scaled2)
df3 = pd.DataFrame(x_scaled3)

# %%

q1 = train[feature_list].quantile(0.25)
q3 = train[feature_list].quantile(0.75)
iqr = q3-q1
fence_low  = q1-1.5*iqr
fence_high = q3+1.5*iqr

train.loc[(train['B1_mean'] > fence_low) & (train['B1_mean'] < fence_high)]


# %%
features = feature_list.tolist()

def remove_outlier(df_in, col_names):
    for col_name in col_names:
        q1 = df_in[col_name].quantile(0.25)
        q3 = df_in[col_name].quantile(0.75)
        iqr = q3-q1 #Interquartile range
        fence_low  = q1-1.5*iqr
        fence_high = q3+1.5*iqr
        df_in = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_in

test = remove_outlier(train, features)


# %%
train = test.copy()

# %%
#distribution of data
_df = train[train.Year == 2016][feature_list.tolist()]
_df2 = train[train.Year == 2017][feature_list.tolist()]
_df3 = train[train.Year == 2018][feature_list.tolist()]


sns.distplot(df, hist = False, kde = True,kde_kws = {'shade': True, 'linewidth': 3}, label = '2016')
sns.distplot(df2, hist = False, kde = True,kde_kws = {'shade': True, 'linewidth': 3}, label = '2017')
sns.distplot(df3, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = '2018')

plt.legend(labels=["2016","2017","2018"])
plt.savefig('distribution_rawbands.png')
# %%
test
# %%
features = ['NDVI_mean','REIP_mean','NDRE_mean']
_2016 = train[train.Year == 2016][features]
_2017 = train[train.Year == 2017][features]
_2018 = train[train.Year == 2018][features]
_2018.drop( _2018.head(14).index, axis=0,inplace=True)
_2017.drop( _2017.head(14).index, axis=0,inplace=True)

_2018.reset_index(inplace=True, drop=True)
_2017.reset_index(inplace=True, drop=True)
_2016.reset_index(inplace=True, drop=True)

# %%
number= 4
_2016.columns=["B"+str(i)+"_2016" for i in range(1, number)]
_2017.columns=["B"+str(i)+"_2017" for i in range(1, number)]
_2018.columns=["B"+str(i)+"_2018" for i in range(1, number)]

# %%
df = pd.concat([_2016, _2017, _2018], axis=1, join='outer')

# %%

svm = sns.heatmap(df.corr())
 
plt.savefig('index_conf.png', dpi=400)
# %%
df
# %%
from scipy import stats
stats.zscore(train[features])
# %%
