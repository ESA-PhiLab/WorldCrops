import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from typing import Optional
from selfsupervised.processing.utils import *
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from .TimeSeriesDataSet import *

class BavariaDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size = 32, num_workers = 2, experiment='Experiment1', **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.experiment = experiment
        
        self.data = pd.read_excel(self.data_dir)
        #list with selected features "Bands 1-13"
        self.feature_list = ['B4_mean','B5_mean','B6_mean','B7_mean','B8_mean','B8A_mean','B9_mean','B11_mean','B12_mean']#self.data.columns[self.data.columns.str.contains('B')].tolist()
        # only NDVI['B3_mean','B4_mean','B5_mean','B6_mean','B7_mean','B8_mean','B8A_mean','B11_mean','B12_mean']#
        #self.feature_list = self.data.columns[self.data.columns.str.contains('NDVI')]

        #update parameters with kwargs
        self.__dict__.update(kwargs)
        if len(self.__dict__) != 0:
            for k in kwargs.keys():
                    self.__setattr__(k, kwargs[k])

        #preprocess
        self.data = clean_bavarian_labels(self.data)
        self.data = remove_false_observation(self.data)
        #filter by date
        self.data = self.data[(self.data['Date'] >= "03-30") & (self.data['Date'] <= "08-30")]

        #data sets
        self.train = None
        self.validate = None
        self.test = None

    def experiment1(self):
        '''data from 2016 2017 and 2018'''
        # use the full data set and make train test split
        splitter = GroupShuffleSplit(test_size=.25, n_splits=2, random_state = 0)
        split = splitter.split(self.data, groups=self.data['id'])
        train_inds, test_inds = next(split)
        train = self.data.iloc[train_inds]
        test = self.data.iloc[test_inds]
        return train, test

    def experiment2(self):
        '''data from 2016 and 2017/ test with 2018'''
        test = self.data[self.data.Year == 2018]
        train = self.data[self.data.Year != 2018]
        return train, test

    def experiment3(self):
        '''data from 2016 and 2017 + 5% 2018'''
        #amount of data in % sampled from 2018
        percent = 5
        _train = self.data[self.data.Year != 2018]
        # x percent sample data for 2018
        _2018 = self.data[self.data.Year == 2018]
        samples = pd.DataFrame()

        for i in range(0,6):
            _ids = _2018[_2018.NC == i].id.unique().tolist()
            
            for id in _ids[:percent]:
                sample = _2018[_2018.id == id]
                samples = pd.concat([samples, sample], axis = 0)
                _2018 = _2018.drop(_2018[_2018.id == id].index)

        train = pd.concat([_train,samples],axis = 0)
        test = _2018
        return train, test

    def experiment4(self):
        '''data from 2016 and 2017 + 10% 2018 '''
        #amount of data in % sampled from 2018
        percent = 10
        _train = self.data[self.data.Year != 2018]
        # x percent sample data for 2018
        _2018 = self.data[self.data.Year == 2018]
        samples = pd.DataFrame()

        for i in range(0,6):
            _ids = _2018[_2018.NC == i].id.unique().tolist()
            
            for id in _ids[:percent]:
                sample = _2018[_2018.id == id]
                samples = pd.concat([samples, sample], axis = 0)
                _2018 = _2018.drop(_2018[_2018.id == id].index)

        train = pd.concat([_train,samples],axis = 0)
        test = _2018
        return train, test

    def exp_without1617_5Prozent(self):
        '''data 5% 2018 '''
        #amount of data in % sampled from 2018
        percent = 5

        unlabeled = self.data[self.data.Year != 2018]
        # x percent sample data for 2018
        _2018 = self.data[self.data.Year == 2018]
        samples = pd.DataFrame()

        for i in range(0,6):
            _ids = _2018[_2018.NC == i].id.unique().tolist()
            
            for id in _ids[:percent]:
                sample = _2018[_2018.id == id]
                samples = pd.concat([samples, sample], axis = 0)
                _2018 = _2018.drop(_2018[_2018.id == id].index)

        train = samples
        test = _2018
        return train, test, unlabeled

    def exp_without1617_10Prozent(self):
        '''data 10% 2018 '''
        #amount of data in % sampled from 2018
        percent = 10

        unlabeled = self.data[self.data.Year != 2018]
        # x percent sample data for 2018
        _2018 = self.data[self.data.Year == 2018]
        samples = pd.DataFrame()

        for i in range(0,6):
            _ids = _2018[_2018.NC == i].id.unique().tolist()
            
            for id in _ids[:percent]:
                sample = _2018[_2018.id == id]
                samples = pd.concat([samples, sample], axis = 0)
                _2018 = _2018.drop(_2018[_2018.id == id].index)

        train = samples
        test = _2018
        return train, test, unlabeled

    def setup(self, stage: Optional[str] = None):

        if self.experiment == 'Experiment1':
            train, test = self.experiment1()
            #callback function for dataframe cleaning, interpolation etc.
            def func(df):
                return clean_bavarian_labels(df)

            ts_train = TSDataSet(train, self.feature_list, 'NC', field_id = 'id',  time_steps = 11)
            ts_test = TSDataSet(test, self.feature_list, 'NC', field_id = 'id',  time_steps = 11)
    
        elif self.experiment == 'Experiment2':
            train, test = self.experiment2()
            #callback function for dataframe cleaning, interpolation etc.
            def func(df):
                return clean_bavarian_labels(df)

            ts_train = TSDataSet(train, self.feature_list, 'NC', field_id = 'id',  time_steps = 11)
            ts_test = TSDataSet(test, self.feature_list, 'NC', field_id = 'id',  time_steps = 11)

        elif self.experiment == 'Experiment3':
            train, test = self.experiment3()
            ts_train = TSDataSet(train, self.feature_list, 'NC', field_id = 'id',  time_steps = 11)
            ts_test = TSDataSet(test, self.feature_list, 'NC', field_id = 'id',  time_steps = 11)

        elif self.experiment == 'Experiment4':
            train, test = self.experiment4()
            ts_train = TSDataSet(train, self.feature_list, 'NC', field_id = 'id',  time_steps = 11)
            ts_test = TSDataSet(test, self.feature_list, 'NC', field_id = 'id',  time_steps = 11)

        elif self.experiment == 'Experiment5':
            train, test, _ = self.exp_without1617_5Prozent()
            ts_train = TSDataSet(train, self.feature_list, 'NC', field_id = 'id',  time_steps = 11)
            ts_test = TSDataSet(test, self.feature_list, 'NC', field_id = 'id',  time_steps = 11)

        elif self.experiment == 'Experiment6':
            train, test, _= self.exp_without1617_10Prozent()
            ts_train = TSDataSet(train, self.feature_list, 'NC', field_id = 'id',  time_steps = 11)
            ts_test = TSDataSet(test, self.feature_list, 'NC', field_id = 'id',  time_steps = 11)

        else:
            print('Experiment not definend')

        # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            self.train = ts_train
            self.validate = ts_test

        # Assign test dataset for use in dataloader(s)
        if stage in (None, "test"):
            self.test = ts_test

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)




class AugmentationExperiments(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size = 32, num_workers = 2, experiment='Experiment1', feature = 'B', **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = pd.read_excel(self.data_dir)
        #list with selected features "Bands 1-13"
        self.feature_list = ['B4_mean','B5_mean','B6_mean','B7_mean','B8_mean','B8A_mean','B9_mean','B11_mean','B12_mean']
        #self.data.columns[self.data.columns.str.contains(feature)].tolist()
        self.experiment = experiment

        #update parameters with kwargs
        self.__dict__.update(kwargs)
        if len(self.__dict__) != 0:
            for k in kwargs.keys():
                    self.__setattr__(k, kwargs[k])

        #preprocess
        self.data = clean_bavarian_labels(self.data)
        self.data = remove_false_observation(self.data)
        self.data = self.data[(self.data['Date'] >= "03-30") & (self.data['Date'] <= "08-30")]
        self.data = rewrite_id_CustomDataSet(self.data)
        #add additional ids for augmentation
        #self.data = augment_df(self.data, [2016,2017,2018])

        #data sets
        self.train = None
        self.validate = None
        self.test = None

    def experiment1(self):
        '''data from 2016 only'''
        self.data = self.data[self.data.Year == 2016]
        splitter = GroupShuffleSplit(test_size=.25, n_splits=2, random_state = 0)
        split = splitter.split(self.data, groups=self.data['id'])
        train_inds, test_inds = next(split)
        train = self.data.iloc[train_inds]
        test = self.data.iloc[test_inds]
        return train, test

    def experiment2(self):
        '''train/test split with data from 2016 and 2017'''
        _data = self.data[self.data.Year != 2018].copy()
        splitter = GroupShuffleSplit(test_size=.25, n_splits=2, random_state = 0)
        split = splitter.split(_data, groups=_data['id'])
        train_inds, test_inds = next(split)
        train = _data.iloc[train_inds]
        test = _data.iloc[test_inds]
        return train, test

    def experiment3(self):
        '''train/test split with data from 2016 2017 and 2018'''
        splitter = GroupShuffleSplit(test_size=.25, n_splits=2, random_state = 0)
        split = splitter.split(self.data, groups=self.data['id'])
        train_inds, test_inds = next(split)
        train = self.data.iloc[train_inds]
        test = self.data.iloc[test_inds]
        return train, test

    def experiment4(self):
        '''train from 2016 and 2017 + 5% 2018 / test with rest 2018'''
        #amount of data in % sampled from 2018
        percent = 5
        _train = self.data[self.data.Year != 2018].copy()
        # x percent sample data for 2018
        _2018 = self.data[self.data.Year == 2018].copy()
        samples = pd.DataFrame()

        for i in range(0,6):
            _ids = _2018[_2018.NC == i].id.unique().tolist()
            
            for id in _ids[:percent]:
                sample = _2018[_2018.id == id]
                samples = pd.concat([samples, sample], axis = 0)
                _2018 = _2018.drop(_2018[_2018.id == id].index)

        train = pd.concat([_train,samples],axis = 0)
        test = _2018
        return train, test

    def experiment5(self):
        '''train from 2016 and 2017 + 10% 2018 / test with rest 2018 '''
        #amount of data in % sampled from 2018
        percent = 10
        _train = self.data[self.data.Year != 2018].copy()
        # x percent sample data for 2018
        _2018 = self.data[self.data.Year == 2018].copy()
        samples = pd.DataFrame()

        for i in range(0,6):
            _ids = _2018[_2018.NC == i].id.unique().tolist()
            
            for id in _ids[:percent]:
                sample = _2018[_2018.id == id]
                samples = pd.concat([samples, sample], axis = 0)
                _2018 = _2018.drop(_2018[_2018.id == id].index)

        train = pd.concat([_train,samples],axis = 0)
        test = _2018
        return train, test

    def exp_without1617_5Prozent(self):
        '''data 5% 2018 '''
        #amount of data in % sampled from 2018
        percent = 5

        unlabeled = self.data[self.data.Year != 2018]
        # x percent sample data for 2018
        _2018 = self.data[self.data.Year == 2018]
        samples = pd.DataFrame()

        for i in range(0,6):
            _ids = _2018[_2018.NC == i].id.unique().tolist()
            
            for id in _ids[:percent]:
                sample = _2018[_2018.id == id]
                samples = pd.concat([samples, sample], axis = 0)
                _2018 = _2018.drop(_2018[_2018.id == id].index)

        train2 = pd.concat([unlabeled,samples],axis = 0)
        train = samples
        test = _2018
        return train, test, train2

    def exp_without1617_10Prozent(self):
        '''data 10% 2018 '''
        #amount of data in % sampled from 2018
        percent = 10

        unlabeled = self.data[self.data.Year != 2018]
        # x percent sample data for 2018
        _2018 = self.data[self.data.Year == 2018]
        samples = pd.DataFrame()

        for i in range(0,6):
            _ids = _2018[_2018.NC == i].id.unique().tolist()
            
            for id in _ids[:percent]:
                sample = _2018[_2018.id == id]
                samples = pd.concat([samples, sample], axis = 0)
                _2018 = _2018.drop(_2018[_2018.id == id].index)

        train2 = pd.concat([unlabeled,samples],axis = 0)
        train = samples
        test = _2018
        return train, test, train2 



    def setup(self, stage: Optional[str] = None):

        if self.experiment == 'Experiment1':
            train ,test = self.experiment1()
            ts_data = CropInvarianceAug(train, self.feature_list, size=10000, time_steps = 11)

        elif self.experiment == 'Experiment2':
            train ,test = self.experiment2()
            ts_data = TSAugmented(train, factor=8, feature_list=self.feature_list, time_steps = 11)

        elif self.experiment == 'Experiment3':
            train, _ = self.experiment3()
            ts_data = CropInvarianceAug(train, self.feature_list, size=10000, time_steps = 11)

        elif self.experiment == 'Experiment4':
            ''' Train invariance between years independent of crop type '''
            train = self.data
            #ts_data = YearInvarianceAug(train, self.feature_list, size=10000)

        elif self.experiment == 'Experiment5':
            ''' Train invariance between crops for 2016/2017 '''
            train = self.data[self.data.Year != 2018].copy()
            ts_data = CropInvarianceAug(train, self.feature_list, size=10000, time_steps = 11)

        elif self.experiment == 'Experiment6':
            ''' Train invariance between crops for 2016/2017 + 5% from 2018 '''
            train, _ = self.experiment4()
            ts_data = CropInvarianceAug(train, self.feature_list, size=10000, time_steps = 11)

        elif self.experiment == 'Experiment7':
            ''' Train invariance between crops for 2016/2017 + 10% from 2018 '''
            train, _ = self.experiment5()
            ts_data = CropInvarianceAug(train, self.feature_list, size=10000, time_steps = 11)

        elif self.experiment == 'Experiment8':
            ''' Train invariance with all data '''
            train, _ = self.experiment3()
            ts_data = TSAugmented(train, factor=8, feature_list=self.feature_list, time_steps = 11)

        elif self.experiment == 'Experiment9':
            ''' Train for 2016/2017 '''
            train = self.data[self.data.Year != 2018].copy()
            ts_data = TSAugmented(train, factor=8, feature_list=self.feature_list, time_steps = 11)

        elif self.experiment == 'Experiment10':
            ''' Train for 2016/2017 + 5% from 2018 '''
            train, test = self.experiment4()
            ts_data = TSAugmented(train, factor=8, feature_list=self.feature_list, time_steps = 11)

        elif self.experiment == 'Experiment11':
            ''' Train for 2016/2017 + 10% from 2018 '''
            train, test = self.experiment5()
            ts_data = TSAugmented(train, factor=8, feature_list=self.feature_list, time_steps = 11)
        
        elif self.experiment == 'Experiment12':
            train, test = self.experiment3()
            ts_data = DriftNoiseAug(train, factor=8, feature_list=self.feature_list, time_steps = 11)

            a = train.mean()[self.feature_list].to_numpy()
            b = test.mean()[self.feature_list].to_numpy()
            diff = b-a
            #ts_data = Shift_TS(train, factor=8, diff = diff, feature_list = self.feature_list, time_steps = 11)

        elif self.experiment == 'Experiment13':
            train = self.data[self.data.Year != 2018].copy()
            test = self.data[self.data.Year == 2018].copy()

            ts_data = DriftNoiseAug(train, factor=8, feature_list=self.feature_list, time_steps = 11)
            a = train.mean()[self.feature_list].to_numpy()
            b = test.mean()[self.feature_list].to_numpy()
            diff = b-a
            #ts_data = Shift_TS(train, factor=8, diff = diff, feature_list = self.feature_list, time_steps = 11)

        elif self.experiment == 'Experiment14':
            train, test = self.experiment4()
            ts_data = DriftNoiseAug(train, factor=8, feature_list=self.feature_list, time_steps = 11)
            a = train.mean()[self.feature_list].to_numpy()
            b = test.mean()[self.feature_list].to_numpy()
            diff = b-a
            #ts_data = Shift_TS(train, factor=8, diff = diff, feature_list = self.feature_list, time_steps = 11)

        elif self.experiment == 'Experiment15':
            ''' Train for 2016/2017 + 10% from 2018 '''
            train, test = self.experiment5()
            ts_data = DriftNoiseAug(train, factor=8, feature_list=self.feature_list, time_steps = 11)
            a = train.mean()[self.feature_list].to_numpy()
            b = test.mean()[self.feature_list].to_numpy()
            diff = b-a
            #ts_data = Shift_TS(train, factor=8, diff = diff, feature_list = self.feature_list, time_steps = 11)

        elif self.experiment == 'Experiment16':
            ''' Train invariance between crops for 2016/2017 + 5% from 2018 '''
            train, _, unlabeled = self.exp_without1617_5Prozent()
            ts_data = CropInvarianceAug(train, self.feature_list, size=10000, time_steps = 11)

        elif self.experiment == 'Experiment17':
            train, _, unlabeled = self.exp_without1617_10Prozent()
            ts_data = CropInvarianceAug(train, self.feature_list, size=10000, time_steps = 11)

        elif self.experiment == 'Experiment18':
            train, _, unlabeled = self.exp_without1617_5Prozent()
            ts_data = DriftNoiseAug(unlabeled, factor=8, feature_list=self.feature_list, time_steps = 11)

        elif self.experiment == 'Experiment19':
            train, _, unlabeled = self.exp_without1617_10Prozent()
            ts_data = DriftNoiseAug(unlabeled, factor=8, feature_list=self.feature_list, time_steps = 11)


        else:
            print('Experiment not definend')

        
       # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            self.train = ts_data
            self.validate = ts_data


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)







