# %%
# compare the crop type classification of RF and SimSiam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split

from selfsupervised.data.croptypes.datamodules import (AugmentationExperiments,
                                                       BavariaDataModule)
from selfsupervised.processing import utils

utils.seed_torch()

with open("../../config/croptypes/param_pretraining.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

batch_size_pre = cfg["pretraining"]['batch_size']
batch_size_fine = cfg["finetuning"]['batch_size']
lr_fine = cfg["finetuning"]['learning_rate']

# parameter for SimSiam
num_ftrs = cfg["pretraining"]['num_ftrs']
proj_hidden_dim = cfg["pretraining"]['proj_hidden_dim']
pred_hidden_dim = cfg["pretraining"]['pred_hidden_dim']
out_dim = cfg["pretraining"]['out_dim']
lr_pre = cfg["pretraining"]['learning_rate']
epochs_pre = cfg["pretraining"]['epochs']
_gpu = cfg["pretraining"]['gpus']
# %%

# %%
# load data for bavaria
# experiment with train/test split for all data
dm_bavaria = BavariaDataModule(
    data_dir='../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx',
    batch_size=batch_size_pre,
    num_workers=4,
    experiment='Experiment1')

# data for invariance between crops
dm_crops1 = AugmentationExperiments(
    data_dir='../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx',
    batch_size=batch_size_pre,
    num_workers=4,
    experiment='Experiment3')

# %%
# create artificail sin cos dataset with 3 classes
seed = 12345512
np.random.seed(seed)

n = 14
x_data = np.linspace(0, 14, num=n)
y_data = np.cos(x_data)
y_data2 = np.sin(x_data)
y_data3 = np.sin(x_data) + 0.3

# %%
plt.plot(y_data)
plt.plot(y_data2)
plt.plot(y_data3)
# %%

X_train, X_test, y_train, y_test = train_test_split(x_data,
                                                    y_data,
                                                    test_size=0.2,
                                                    random_state=42)

# %%
# setup DataModule
# load data for bavaria
bavaria_data = pd.read_excel(
    "../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx")
bavaria_data = utils.clean_bavarian_labels(bavaria_data)
bavaria_data = utils.remove_false_observation(bavaria_data)
bavaria_data = bavaria_data[(bavaria_data['Date'] >= "03-30")
                            & (bavaria_data['Date'] <= "08-30")]
bavaria_data = utils.rewrite_id_CustomDataSet(bavaria_data)
# %%

bavaria_data

# %%
class HopfieldSampling(nn.Module):
    '''
   Mapping of N samples from year X onto n samples from year Y.
   '''

    def __init__(self, t_steps, channels) -> None:
        super(HopfieldSampling, self).__init__()
        self.unflat = nn.Unflatten(1, (t_steps, channels))
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, base, sample, beta):
        '''
       BASE are examples of the unknown year (e.g. 2018) to which
       the SAMPLE (e.g. 2017) is projected to produce the MAPPED_SAMPLE 
       (e.g. fake 2018).
       Larger beta causes stronger mixing.
       DIMENSION:
       base:   [#Examples, TimeSteps, Channels]
       sample: [#Samples, TimeSteps, Channels]
       mapped_sample: [#Samples, TimeSteps, Channels]
       '''

        base = torch.flatten(base, start_dim=1, end_dim=2)
        sample = torch.flatten(sample, start_dim=1, end_dim=2)
        Q = sample
        K = base
        V = base
        p = torch.einsum('ac,dc->ad', Q, K)
        p = self.Softmax(beta * p)
        print(p)
        mapped_sample = torch.einsum('ac,cb->ab', p, V)
        mapped_sample = self.unflat(mapped_sample)
        return mapped_sample


# H = HopfieldSampling(t_steps, channels)

# %% How to use.
beta = 0.3
t_steps = 14
channels = 13

test = HopfieldSampling(t_steps, channels)

base = torch.zeros((3, 14, 13))  # 3 examples of unknown year
sample = torch.zeros((10, 14, 13))  # 10 samples of known year
mapped_sample = test(base, sample, beta)  # creates 10 fake samples

print(mapped_sample.shape)
# %%
feature = 'NDVI_mean'
time_steps = 11
year = 2018


def getCroptype(data, feature, type, year, time_steps):
    features = [feature]
    target = data[(data.Year == year) & (data.NC == type)]
    base = target[features].to_numpy().reshape(len(target['id'].unique()),
                                               time_steps, len(features))
    # choosen_ids =
    # [ random.choice(bavaria_data[(bavaria_data.Year == year)
    # & (bavaria_data.NC == type)].id.to_list())  for x in range(5) ]
    # base = bavaria_data.loc[bavaria_data['id'].isin(choosen_ids)]
    # base = base[feature].to_numpy().reshape( len(base['id'].unique()),
    # time_steps , len(features) )
    return base


def getall2018(data, feature, year, time_steps):
    features = [feature]
    target = data[(data.Year == year)]
    base = target[features].to_numpy().reshape(len(target['id'].unique()),
                                               time_steps, len(features))
    return base


# %%
croptype = 1

known = bavaria_data[(bavaria_data.Year == 2016)
                     & (bavaria_data.NC == croptype)][feature].to_numpy()
fields_known = len(
    bavaria_data[(bavaria_data.Year == 2016)
                 & (bavaria_data.NC == croptype)]['id'].unique())
known = known.reshape(fields_known, 11, 1)
# known = known[6,:,:].reshape( 1, 11 , 1 )

base = getCroptype(bavaria_data, feature, croptype, year, time_steps)
H = HopfieldSampling(time_steps, 1)
# %%
mapped_sample = H(torch.from_numpy(base), torch.from_numpy(known), 5)

for i in range(4):
    plt.plot(mapped_sample[i], 'r')
    plt.plot(base[i], 'b')
    plt.plot(known[i], 'y')
# %%
base = getCroptype(bavaria_data, feature, croptype, year, time_steps)
for i in range(90):
    plt.plot(base[i], 'r')

# %%
# compare against all from 2018
base = getall2018(bavaria_data, feature, year, time_steps)
mapped_sample = H(torch.from_numpy(base), torch.from_numpy(known), 5)
for i in range(4):
    plt.plot(mapped_sample[i], 'r')
    plt.plot(base[i], 'b')
    plt.plot(known[i], 'y')
# %%


# %%
