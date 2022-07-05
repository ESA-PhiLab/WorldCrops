#%%
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import pandas as pd
import random
from tsaug import AddNoise, Convolve, Crop, Drift, Dropout, Pool, Quantize, Resize, Reverse, TimeWarp
from random import randrange, uniform

class OwnAugmentation():

    def jitter(x, sigma=0.03):
        # https://arxiv.org/pdf/1706.00527.pdf
        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

    def scaling(x, sigma=0.1):
        # https://arxiv.org/pdf/1706.00527.pdf
        factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[1]))
        return np.multiply(x, factor[:,:])

    def constant_reflectance_change(x, min = 0, max = 10 ):
        '''
        Shift band time series 
        :param min/max: minimum and maximum value to apply
        :param x: numpy array with band values
        '''
        shift = randrange(int(min), int(max))
        return x + shift


    def bands_reflectance_change(x, band_list):
        '''
        Shift band time series for each band
        :param band_list: contains the shift from one data set to another (e.g. 2016 -> 2017) for each band
        :param x: numpy array with band values
        '''

        for band in range(0,len(band_list)):
            diff = int(band_list[band])
            if diff <0:
                shift = randrange(diff, 0)
            else:
                shift = randrange(0, diff)

            x[:,band] = x[:,band] + shift

        return x

    def bands_noise(x, bands_noise):
        '''
        Add cloud noise for each band and a time point
        :param bands_noise: noise values for each band
        :param x: numpy array with band values
        '''
        _timestep = randrange(0, x.shape[0])
        for band in range(0,len(bands_noise)):
            x[_timestep,band] = x[_timestep,band] + bands_noise[band]
        return x

    def constant_noise(x, band_noise):
        '''
        Add same noise for all bands
        :param bands_noise: noise value
        :param x: numpy array with band values
        '''

        _timestep = randrange(0, x.shape[0])
        x[_timestep,:] = band_noise
        return x


my_augmenter = (
     #TimeWarp() * 5  # random time warping 5 times in parallel
     #+ Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
     Drift(max_drift=(0.1, 0.5)),   # with 80% probability, random drift the signal up to 10% - 50%
     AddNoise(scale=0.1)
     #+ Reverse() @ 0.5  # with 50% probability, reverse the sequence 
)

class AugmentationSampling():
    '''Obtain mean and std for each timestep from dataset and draw augmentation from that.
       REQUIRES: data[year,type,channel,timestep,samples]
    '''
    def __init__(self, data) -> None:
        self.years = data.shape[0]
        self.types = data.shape[1]
        self.channels = data.shape[2]
        self.time_steps = data.shape[3]
        self.mu = torch.zeros((self.years, self.types, self.channels, self.time_steps))
        self.std = torch.zeros((self.years, self.types, self.channels, self.time_steps))
        for y in range(self.years):
            for f in range(self.types):
                for c in range(self.channels):
                    for t in range(self.time_steps):
                        self.mu[y,f,c,t] = torch.mean(torch.tensor(data[y,f,c,t,:]))
                        self.std[y,f,c,t] = torch.std(torch.tensor(data[y,f,c,t,:]))

    def create_augmentation(self, year, type, n_samples):
        if str(year)=='2016':
            year = 0
        if str(year)=='2017':
            year = 1
        if str(year)=='2018':
            year = 2
        samples = torch.zeros((n_samples, self.channels, self.time_steps))
        for n in range(n_samples):
            for c in range(self.channels):
                for t in range(self.time_steps):
                    # print(year,n,c,t)
                    # print(self.mu.shape, self.std.shape)
                    samples[n,c,t] = torch.normal(mean=self.mu[year,type,c,t], std=self.std[year,type,c,t])
        return samples


class TSAugmented(Dataset):
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
        n_years = len(data.Year.unique())
        self.n_years = n_years
        n_channels = len(self.feature_list)
        n_tsteps = self.time_steps
        n_samples = 100
        entries = data.shape[0]
        # : Required data format
        data_sorted = np.zeros((n_years,n_features,n_channels,n_tsteps,n_samples))
        for y in range(n_years):
            for m in range(n_features):
                cnt = 0
                fcnt = 0
                for n in range(entries):
                    if(str(temp_data[n,-1])==str(data.Year.unique()[y])):
                        if(temp_data[n,3]==m):
                            if (cnt==14):
                                fcnt += 1
                                cnt = 0
                            data_sorted[y,m,:n_channels,cnt,fcnt] = temp_data[n,4:17] 
                            cnt += 1

        # :: Initialize statistical augmentation object
        self.aug_sample = AugmentationSampling(data_sorted)
        # : Usage: self.aug_sample.create_augmentation([year], [type], [n_samples])
        # [year] = [2016, 2017, 2018] OR [0, 1, 2]
        # : Example creates 2 samples for crop type 0 using the statistics obtained for 2016
        # aug_samples = self.aug_sample.create_augmentation(2016,0,2)
        # OR IDENTICALLY:
        # aug_samples = self.aug_sample.create_augmentation(1,0,2)

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

        # : Augmentation 1
        year = int(np.rint(self.n_years * np.random.rand())) - 1       # 0: 2016, 1: 2017, 2: 2018
        aug_samples = self.aug_sample.create_augmentation(year,y.item(),1)
        aug_x1 = aug_samples[0].permute(1,0)
        # : Augmentation 2
        year = int(np.rint(self.n_years * np.random.rand())) - 1       # 0: 2016, 1: 2017, 2: 2018
        aug_samples = self.aug_sample.create_augmentation(year,y.item(),1)
        aug_x2 = aug_samples[0].permute(1,0)

        torchx = self.x2torch(x)
        torchy = self.y2torch(y)
        # ::: Training on purely statistically generated data pairs (aug_x1, aug_x2)
        return (aug_x1, aug_x2), torchx, torchy #, torch.tensor(field_id, dtype=torch.long)
        # ::: Training on a mix of real (torchx) samples and statistically generated samples (aug_x1)
        # return (aug_x1, torchx), torchx, torchy #, torch.tensor(field_id, dtype=torch.long)
        
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


class CropInvarianceAug(Dataset):
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
        self.labels = np.zeros((self.size, 1 ))
        self.sampleData()


    def sampleData(self):
        try:
            for idx in range(self.size):
                ts1, ts2, y = self.get_X1_X2(self.df, self.feature_list)
                self.augmented[idx,0] = ts1
                self.augmented[idx,1] = ts2
                self.labels[idx,0] = y
        except:
            print('Error in data generation:', ts1.shape, ts2.shape, idx)

        
    def get_X1_X2(self, data, features):
        '''Returns two different timeseries for the same crop
        '''
        random_field = random.choice(data.id.unique())
        random_crop = random.choice(data.NC.unique())

        #two different years
        #year_list = data.Year.unique().tolist()
        #random_year1 = random.choice(year_list)
        #year_list.remove( random_year1 )
        #random_year2 = random.choice(year_list)

        #choose same crop but from different years
        field_id1 = random.choice(data[data.NC == random_crop].id.unique())
        field_id2 = random.choice(data[data.NC == random_crop].id.unique())

        X1 = data[data.id == field_id1][features].to_numpy()
        X2 = data[data.id == field_id2][features].to_numpy()
        #X1 = OwnAugmentation.constant_noise(X1, 7000)
        #X2 = OwnAugmentation.constant_noise(X2, 7000)

        return X1, X2, random_crop

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x1 = self.augmented[idx,0]
        x2 = self.augmented[idx,1]
        y = self.labels[idx,0]

        #augmentation based on different years
        aug_x1 = self.x2torch(x1)
        aug_x2 = self.x2torch(x2)
        y = self.y2torch(y)

        #None torch values for x,y
        x = torch.from_numpy(np.array(0)).type(torch.FloatTensor)
        #y = torch.from_numpy(y).type(torch.FloatTensor)

        return (aug_x1, aug_x2), x, y

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

class CropInvarianceAug2(Dataset):
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
        self.labels = np.zeros((self.size, 1 ))
        self.sampleData()


    def sampleData(self):
        try:
            for idx in range(self.size):
                ts1, ts2, y = self.get_X1_X2(self.df, self.feature_list)
                self.augmented[idx,0] = ts1
                self.augmented[idx,1] = ts2
                self.labels[idx,0] = y
        except:
            print('Error in data generation:', ts1.shape, ts2.shape, idx)

        
    def get_X1_X2(self, data, features):
        '''Returns two different timeseries for the same crop
        '''
        random_field = random.choice(data.id.unique())
        random_crop = random.choice(data.NC.unique())

        #two different years
        #year_list = data.Year.unique().tolist()
        #random_year1 = random.choice(year_list)
        #year_list.remove( random_year1 )
        #random_year2 = random.choice(year_list)

        #choose same crop but from different years
        field_id1 = random.choice(data[data.NC == random_crop].id.unique())
        field_id2 = random.choice(data[data.NC == random_crop].id.unique())

        X1 = data[data.id == field_id1][features].to_numpy()
        X2 = data[data.id == field_id2][features].to_numpy()
        X1 = OwnAugmentation.constant_noise(X1, 7000)
        X2 = OwnAugmentation.constant_noise(X2, 7000)

        return X1, X2, random_crop

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x1 = self.augmented[idx,0]
        x2 = self.augmented[idx,1]
        y = self.labels[idx,0]

        #augmentation based on different years
        aug_x1 = self.x2torch(x1)
        aug_x2 = self.x2torch(x2)
        y = self.y2torch(y)

        #None torch values for x,y
        x = torch.from_numpy(np.array(0)).type(torch.FloatTensor)
        #y = torch.from_numpy(y).type(torch.FloatTensor)

        return (aug_x1, aug_x2), x, y

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

            
        
class DriftNoiseAug(Dataset):

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

        aug_type = random.choice([0,1])

        if aug_type == 0:
            aug_x1 = x
            aug_x2 = Drift(max_drift=0.1, n_drift_points=2).augment(x)
        else:
            aug_x1 = x
            aug_x2 = AddNoise(scale=0.02).augment(x)

        torchx = self.x2torch(x)
        torchy = self.y2torch(y)

        return (self.x2torch(aug_x1), self.x2torch(aug_x2)), torchx, torchy #, torch.tensor(field_id, dtype=torch.long)

        
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


class Shift_TS(Dataset):

    def __init__(self, data, factor=1, feature_list = [], target_col = 'NC', field_id = 'id', time_steps = 14, diff=np.array([0]), callback = None):
        self.df = data
        self.factor = factor
        self.df = self.reproduce(data, self.factor)
        self.target_col = target_col
        self.feature_list = feature_list
        self.time_steps = time_steps
        self.diff = diff

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
        if len(self.diff) != len(self.feature_list):
            print("Define max shift for each band in diff")
            return
        
        #reshape to 3D
        #field x T x D
        self.df = self.df.reshape(int(self._fields_amount),self.time_steps, len(self.feature_list))
        self.y = self.y.reshape(int(self._fields_amount),1, self.time_steps)
        self.field_ids = self.field_ids.reshape(int(self._fields_amount),1, self.time_steps)


    def reproduce(self, df, _size):
        ''' reproduce the orginal df with factor X times'''
        newdf = pd.DataFrame()
        if _size > 1:
            for idx in range(_size):
                newdf = pd.concat([newdf, df.copy()], axis=0)
                #print(len(newdf),_size)
            return newdf
        else:
            return df

    def __len__(self):
        return self.df.shape[0]


    def __getitem__(self, idx):
        x = self.df[idx,:,:]
        y = self.y[idx,0,0]
        field_id = self.field_ids[idx,0,0]

        aug_type = 0 #random.choice([0,1])

        if aug_type == 0:
            aug_x1 = OwnAugmentation.bands_reflectance_change(x, self.diff)
            aug_x2 = OwnAugmentation.bands_reflectance_change(x, self.diff)
        if aug_type == 1:
            aug_x1 = OwnAugmentation.constant_noise(x, 7000)
            aug_x2 = OwnAugmentation.constant_noise(x, 7000)

        torchx = self.x2torch(x)
        torchy = self.y2torch(y)

        return (self.x2torch(aug_x1), self.x2torch(aug_x2)), torchx, torchy #, torch.tensor(field_id, dtype=torch.long)

        
    def x2torch(self, x):
        '''return torch for x'''
        return torch.from_numpy(x).type(torch.FloatTensor)

    def y2torch(self, y):
        '''return torch for y'''
        return torch.tensor(y, dtype=torch.long)
