
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, AddNoise
import pandas as pd
import random

class OwnAugmentation():

  def jitter(x, sigma=0.03):
      # https://arxiv.org/pdf/1706.00527.pdf
      return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

  def scaling(x, sigma=0.1):
      # https://arxiv.org/pdf/1706.00527.pdf
      factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[1]))
      return np.multiply(x, factor[:,:])

class AugmentationSampling():
    '''Obtain mean and std per timestep from dataset and draw augmentation from that.
       REQUIRES: data[type][channel,timestep,samples]
    '''
    def __init__(self, data) -> None:
        self.channels = data.shape[0]
        self.types = data.shape[1]
        self.time_steps = data.shape[2]
        self.mu = torch.zeros((self.types, self.channels, self.time_steps))
        self.std = torch.zeros((self.types, self.channels, self.time_steps))
        for f in range(self.types):
            for c in range(self.channels):
                for t in range(self.time_steps):
                    self.mu[f,c,t] = torch.mean(torch.tensor(data[f][c,:,t]))
                    self.std[f,c,t] = torch.std(torch.tensor(data[f][c,:,t]))

    def create_augmentation(self, type, n_samples):
        samples = torch.zeros((n_samples, self.channels,self.time_steps))
        for n in range(n_samples):
            for c in range(self.channels):
                for t in range(self.time_steps):
                    samples[n,c,t] = torch.normal(mean=self.mu[type,c,t], std=self.std[type,c,t])
        return samples


class Bootstrapping():
    '''Draw sample and augmentation from original dataset.'''
    pass


class BootstrapResampling():
    '''? Obtain mean and std per timestepd based on bootstrapping and draw augmentation from that.'''
    pass


my_augmenter = (
     #TimeWarp() * 5  # random time warping 5 times in parallel
     #+ Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
     Drift(max_drift=(0.1, 0.5)),   # with 80% probability, random drift the signal up to 10% - 50%
     AddNoise(scale=0.1)
     #+ Reverse() @ 0.5  # with 50% probability, reverse the sequence 
)


class TimeSeriesAugmented(Dataset):
    '''
    :param data: dataset of type pandas.DataFrame
    :param target_col: targeted column name
    :param feature_list: list with target features
    '''

    def __init__(self, data, feature_list, target_col):
        self.xy = data
        self.target_col = target_col
        self.feature_list = feature_list

    def __len__(self):
        return len(self.xy.id.unique())

    def __getitem__(self, field_idx):

        x = self.xy[self.xy.id == field_idx][self.feature_list].values
        y = self.xy[self.xy.id == field_idx][self.target_col].values

        torchx = self.x2torch(x)
        torchy = self.y2torch(y)

        #augmentation with jitter and scaling
        aug_x1 = self.x2torch(OwnAugmentation.jitter(x))
        aug_x2 = self.x2torch(OwnAugmentation.scaling(x))

        return (aug_x1, aug_x2), torchx, torchy
        #return torchx, torchy
        
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
        y = y[1]
        return torch.tensor(y, dtype=torch.long)

class TimeSeriesPhysical(Dataset):
    '''
    :param data: dataset of type pandas.DataFrame
    :param target_col: targeted column name
    :param feature_list: list with target features
    '''

    def __init__(self, data, feature_list, target_col):
        self.xy = data
        self.target_col = target_col
        self.feature_list = feature_list

    def __len__(self):
        return len(self.xy.id.unique())

    def get_other_years(self,currentyear, yearlist):
        import random
        yearlist.remove(currentyear)
        output = random.sample(yearlist, len(yearlist))
        return output[0],output[1]

    def __getitem__(self, field_idx):

        x = self.xy[self.xy.id == field_idx][self.feature_list].values
        y = self.xy[self.xy.id == field_idx][self.target_col].values

        _id1 = self.xy[self.xy.id == field_idx]['id_x1'].values[0]
        _id2 = self.xy[self.xy.id == field_idx]['id_x2'].values[0]
        x1 = self.xy[self.xy.id == int(_id1)][self.feature_list].values
        x2 = self.xy[self.xy.id == int(_id2)][self.feature_list].values

        torchx = self.x2torch(x)
        torchy = self.y2torch(y)

        #augmentation based on different years
        aug_x1 = self.x2torch(x1)
        aug_x2 = self.x2torch(x2)

        return (aug_x1, aug_x2), torchx, torchy
            
        
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
        y = y[1]
        return torch.tensor(y, dtype=torch.long)

class TimeSeriesDataSet(Dataset):
    '''
    :param data: dataset of type pandas.DataFrame
    :param target_col: targeted column name
    :param feature_list: list with target features
    '''

    def __init__(self, data, feature_list, target_col):
        self.xy = data
        self.target_col = target_col
        self.feature_list = feature_list

    def __len__(self):
        return len(self.xy.id.unique())

    def __getitem__(self, field_idx):

        x = self.xy[self.xy.id == field_idx][self.feature_list].values
        y = self.xy[self.xy.id == field_idx][self.target_col].values

        torchx = self.x2torch(x)
        torchy = self.y2torch(y)

        return torchx, torchy
        #return torchx, torchy
        
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
        y = y[1]
        return torch.tensor(y, dtype=torch.long)


class TSDataSet(Dataset):
    '''
    :param data: dataset of type pandas.DataFrame
    :param target_col: targeted column name
    :param field_id: name of column with field ids
    :param feature_list: list with target features
    :param callback: preprocessing of dataframe
    '''
    def __init__(self, data, feature_list = [], target_col = 'NC', field_id = 'id', time_steps = 14, callback = None):
        self.df = data
        self.target_col = target_col
        self.feature_list = feature_list
        self.time_steps = time_steps

        if callback != None:
            self.df = callback(self.df)

        self._fields_amount = len(self.df[field_id].unique())

        #get numpy
        self.y = self.df[self.target_col].values
        self.field_ids = self.df[field_id].values
        self.df = self.df[self.feature_list].values

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


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        x = self.df[idx,:,:]
        y = self.y[idx,0,0]
        field_id = self.field_ids[idx,0,0]

        torchx = self.x2torch(x)
        torchy = self.y2torch(y)
        return torchx, torchy #, torch.tensor(field_id, dtype=torch.long)
        
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


class Test2016(Dataset):
    '''
    :param data: dataset of type pandas.DataFrame
    :param target_col: targeted column name
    :param feature_list: list with target features
    '''

    def __init__(self, data, feature_list = [], target_col = 'NC', field_id = 'id', time_steps = 14, callback = None, size = 0):
        self.df = data
        self.target_col = target_col
        self.feature_list = feature_list
        self.time_steps = time_steps
        self.size = size

        if self.size == 0:
            print('Define data size')
            return
        if callback != None:
            self.df = callback(self.df)

        #numpy with augmented data
        #size x 2 x T x D
        self.augmented = np.zeros((self.size, 2, self.time_steps, len(self.feature_list)))
        self.sampleData()


    def sampleData(self):
        try:
            for idx in range(self.size):
                ts1, ts2 = self.get_X1_X2(self.df, self.feature_list)
                self.augmented[idx,0] = ts1
                self.augmented[idx,1] = ts2
        except:
            print('Error in data generation:', ts1.shape, ts2.shape, idx)

        
    def get_X1_X2(self, data, features):
        '''Returns two different timeseries for the same crop
        '''
        random_field = random.choice(data.id.unique())
        random_crop = random.choice(data.NC.unique())
        #choose random crop and then two random fields from this crop
        field_id1 = random.choice(data[data.NC == random_crop].id.unique())
        field_id2 = random.choice(data[data.NC == random_crop].id.unique())
        return data[data.id == field_id1][features].to_numpy(), data[data.id == field_id2][features].to_numpy()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        x1 = self.augmented[idx,0]
        x2 = self.augmented[idx,1]

        #augmentation based on different years
        aug_x1 = self.x2torch(x1)
        aug_x2 = self.x2torch(x2)

        #None torch values for x,y
        x = torch.from_numpy(np.array(0)).type(torch.FloatTensor)
        y = torch.from_numpy(np.array(0)).type(torch.FloatTensor)

        return (aug_x1, aug_x2), x, y

    def x2torch(self, x):
        '''
        return torch for x
        '''
        #nb_obs, nb_features = self.x.shape
        return torch.from_numpy(x).type(torch.FloatTensor)


            
        
    
