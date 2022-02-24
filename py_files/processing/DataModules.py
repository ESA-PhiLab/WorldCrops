import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from typing import Optional
from .utils import *
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from .TimeSeriesDataSet import *

class BavariaDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size = 32, num_workers = 2, experiment='Experiment1'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.experiment = experiment
        
        self.data = pd.read_excel(self.data_dir)
        #list with selected features "Bands 1-13"
        self.feature_list = self.data.columns[self.data.columns.str.contains('B')].tolist()
        # only NDVI['B3_mean','B4_mean','B5_mean','B6_mean','B7_mean','B8_mean','B8A_mean','B11_mean','B12_mean']#
        #self.feature_list = self.data.columns[self.data.columns.str.contains('NDVI')]

        #preprocess
        self.data = clean_bavarian_labels(self.data)

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

        for j in range(percent):
            #one sample per crop type
            for i in range(0,6):
                id = _2018[(_2018.NC == i)].sample(1).id
                sample = _2018[(_2018.id == id.values[0])]
                #delete row in orginal 2018 data
                _2018 = _2018.drop(_2018[_2018.id == id.values[0]].index)
                samples = pd.concat([samples,sample],axis=0)
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

        for j in range(percent):
            #one sample per crop type
            for i in range(0,6):
                id = _2018[(_2018.NC == i)].sample(1).id
                sample = _2018[(_2018.id == id.values[0])]
                #delete row in orginal 2018 data
                _2018 = _2018.drop(_2018[_2018.id == id.values[0]].index)
                samples = pd.concat([samples,sample],axis=0)
        train = pd.concat([_train,samples],axis = 0)
        test = _2018
        return train, test

    def setup(self, stage: Optional[str] = None):

        if self.experiment == 'Experiment1':
            train, test = self.experiment1()
            #callback function for dataframe cleaning, interpolation etc.
            def func(df):
                return clean_bavarian_labels(df)

            ts_train = TSDataSet(train, self.feature_list, 'NC', field_id = 'id')
            ts_test = TSDataSet(test, self.feature_list, 'NC', field_id = 'id')
    
        elif self.experiment == 'Experiment2':
            train, test = self.experiment2()
            #callback function for dataframe cleaning, interpolation etc.
            def func(df):
                return clean_bavarian_labels(df)

            ts_train = TSDataSet(train, self.feature_list, 'NC', field_id = 'id')
            ts_test = TSDataSet(test, self.feature_list, 'NC', field_id = 'id')

        elif self.experiment == 'Experiment3':
            train, test = self.experiment3()
            ts_train = TSDataSet(train, self.feature_list, 'NC', field_id = 'id')
            ts_test = TSDataSet(test, self.feature_list, 'NC', field_id = 'id')

        elif self.experiment == 'Experiment4':
            train, test = self.experiment4()
            ts_train = TSDataSet(train, self.feature_list, 'NC', field_id = 'id')
            ts_test = TSDataSet(test, self.feature_list, 'NC', field_id = 'id')

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



class DataModule_augmentation(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size = 32, num_workers = 2):
        '''augment between 2016 and 2017 for each crop type'''
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = pd.read_excel(self.data_dir)
        #list with selected features "Bands 1-13"
        self.feature_list = self.data.columns[self.data.columns.str.contains('B')].tolist()
        # only NDVI
        #self.feature_list = self.data.columns[self.data.columns.str.contains('NDVI')]

        #preprocess
        self.data = clean_bavarian_labels(self.data)
        self.data = self.data[self.data.Year != 2018]
        self.data = rewrite_id_CustomDataSet(self.data)
        #add additional ids for augmentation
        self.data = augment_df(self.data, [2016,2017])

        #data sets
        self.train = None
        self.validate = None
        self.test = None

    def setup(self, stage: Optional[str] = None):
        ts_train = TimeSeriesPhysical(self.data, self.feature_list, 'NC')

        # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            self.train = ts_train

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers)


class Augmentation_Crops(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size = 32, num_workers = 2, experiment='Experiment1', feature = 'B'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = pd.read_excel(self.data_dir)
        #list with selected features "Bands 1-13"
        self.feature_list = self.data.columns[self.data.columns.str.contains(feature)]
        self.experiment = experiment

        #preprocess
        self.data = clean_bavarian_labels(self.data)
        self.data = rewrite_id_CustomDataSet(self.data)
        #add additional ids for augmentation
        self.data = augment_df(self.data, [2016,2017,2018])

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
        self.data = self.data[self.data.Year != 2018]
        splitter = GroupShuffleSplit(test_size=.25, n_splits=2, random_state = 0)
        split = splitter.split(self.data, groups=self.data['id'])
        train_inds, test_inds = next(split)
        train = self.data.iloc[train_inds]
        test = self.data.iloc[test_inds]
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
        _train = self.data[self.data.Year != 2018]
        # x percent sample data for 2018
        _2018 = self.data[self.data.Year == 2018]
        samples = pd.DataFrame()

        for j in range(percent):
            #one sample per crop type
            for i in range(0,6):
                id = _2018[(_2018.NC == i)].sample(1).id
                sample = _2018[(_2018.id == id.values[0])]
                #delete row in orginal 2018 data
                _2018 = _2018.drop(_2018[_2018.id == id.values[0]].index)
                samples = pd.concat([samples,sample],axis=0)
        train = pd.concat([_train,samples],axis = 0)
        test = _2018
        return train, test

    def experiment5(self):
        '''train from 2016 and 2017 + 10% 2018 / test with rest 2018 '''
        #amount of data in % sampled from 2018
        percent = 10
        _train = self.data[self.data.Year != 2018]
        # x percent sample data for 2018
        _2018 = self.data[self.data.Year == 2018]
        samples = pd.DataFrame()

        for j in range(percent):
            #one sample per crop type
            for i in range(0,6):
                id = _2018[(_2018.NC == i)].sample(1).id
                sample = _2018[(_2018.id == id.values[0])]
                #delete row in orginal 2018 data
                _2018 = _2018.drop(_2018[_2018.id == id.values[0]].index)
                samples = pd.concat([samples,sample],axis=0)
        train = pd.concat([_train,samples],axis = 0)
        test = _2018
        return train, test



    def setup(self, stage: Optional[str] = None):

        if self.experiment == 'Experiment1':
            train ,test = self.experiment1()
            ts_data = CropInvarianceAug(train, self.feature_list.tolist(), size=10000)

        elif self.experiment == 'Experiment2':
            train ,test = self.experiment2()
            ts_data = TSAugmented(train, factor=2, feature_list=self.feature_list.tolist())

        elif self.experiment == 'Experiment3':
            train, _ = self.experiment3()
            ts_data = CropInvarianceAug(train, self.feature_list.tolist(), size=10000)

        elif self.experiment == 'Experiment4':
            ''' Train invariance between years independent of crop type '''
            train = self.data
            ts_data = YearInvarianceAug(train, self.feature_list.tolist(), size=10000)

        elif self.experiment == 'Experiment5':
            ''' Train invariance between crops for 2016/2017 '''
            train = self.data[self.data.Year != 2018]
            ts_data = CropInvarianceAug(train, self.feature_list.tolist(), size=10000)

        elif self.experiment == 'Experiment6':
            ''' Train invariance between crops for 2016/2017 + 5% from 2018 '''
            train, _ = self.experiment4()
            ts_data = CropInvarianceAug(train, self.feature_list.tolist(), size=10000)

        elif self.experiment == 'Experiment7':
            ''' Train invariance between crops for 2016/2017 + 10% from 2018 '''
            train, _ = self.experiment5()
            ts_data = CropInvarianceAug(train, self.feature_list.tolist(), size=10000)

        elif self.experiment == 'Experiment8':
            ''' Train invariance with all data '''
            train, _ = self.experiment3()
            ts_data = TSAugmented(train, factor=5, feature_list=self.feature_list.tolist())

        elif self.experiment == 'Experiment9':
            ''' Train for 2016/2017 '''
            train = self.data[self.data.Year != 2018]
            ts_data = TSAugmented(train, factor=5, feature_list=self.feature_list.tolist())

        elif self.experiment == 'Experiment10':
            ''' Train for 2016/2017 + 5% from 2018 '''
            train, _ = self.experiment4()
            ts_data = TSAugmented(train, factor=5, feature_list=self.feature_list.tolist())

        elif self.experiment == 'Experiment11':
            ''' Train for 2016/2017 + 10% from 2018 '''
            train, _ = self.experiment5()
            ts_data = TSAugmented(train, factor=5, feature_list=self.feature_list.tolist())

        else:
            print('Experiment not definend')

        
       # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            self.train = ts_data
            self.validate = ts_data
            print(len(self.train))


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers)

    #def val_dataloader(self):
    #    return DataLoader(self.validate, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)

class AugmentationExperiments(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size = 32, num_workers = 2, experiment='Experiment1', feature = 'B'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = pd.read_excel(self.data_dir)
        #list with selected features "Bands 1-13"
        self.feature_list = self.data.columns[self.data.columns.str.contains(feature)]
        self.experiment = experiment

        #preprocess
        self.data = clean_bavarian_labels(self.data)
        self.data = rewrite_id_CustomDataSet(self.data)
        #add additional ids for augmentation
        self.data = augment_df(self.data, [2016,2017,2018])

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
        self.data = self.data[self.data.Year != 2018]
        splitter = GroupShuffleSplit(test_size=.25, n_splits=2, random_state = 0)
        split = splitter.split(self.data, groups=self.data['id'])
        train_inds, test_inds = next(split)
        train = self.data.iloc[train_inds]
        test = self.data.iloc[test_inds]
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
        _train = self.data[self.data.Year != 2018]
        # x percent sample data for 2018
        _2018 = self.data[self.data.Year == 2018]
        samples = pd.DataFrame()

        for j in range(percent):
            #one sample per crop type
            for i in range(0,6):
                id = _2018[(_2018.NC == i)].sample(1).id
                sample = _2018[(_2018.id == id.values[0])]
                #delete row in orginal 2018 data
                _2018 = _2018.drop(_2018[_2018.id == id.values[0]].index)
                samples = pd.concat([samples,sample],axis=0)
        train = pd.concat([_train,samples],axis = 0)
        test = _2018
        return train, test

    def experiment5(self):
        '''train from 2016 and 2017 + 10% 2018 / test with rest 2018 '''
        #amount of data in % sampled from 2018
        percent = 10
        _train = self.data[self.data.Year != 2018]
        # x percent sample data for 2018
        _2018 = self.data[self.data.Year == 2018]
        samples = pd.DataFrame()

        for j in range(percent):
            #one sample per crop type
            for i in range(0,6):
                id = _2018[(_2018.NC == i)].sample(1).id
                sample = _2018[(_2018.id == id.values[0])]
                #delete row in orginal 2018 data
                _2018 = _2018.drop(_2018[_2018.id == id.values[0]].index)
                samples = pd.concat([samples,sample],axis=0)
        train = pd.concat([_train,samples],axis = 0)
        test = _2018
        return train, test



    def setup(self, stage: Optional[str] = None):

        if self.experiment == 'Experiment1':
            train ,test = self.experiment1()
            ts_data = CropInvarianceAug(train, self.feature_list.tolist(), size=10000)

        elif self.experiment == 'Experiment2':
            train ,test = self.experiment2()
            ts_data = TSAugmented(train, factor=2, feature_list=self.feature_list.tolist())

        elif self.experiment == 'Experiment3':
            train, _ = self.experiment3()
            ts_data = CropInvarianceAug(train, self.feature_list.tolist(), size=10000)

        elif self.experiment == 'Experiment4':
            ''' Train invariance between years independent of crop type '''
            train = self.data
            ts_data = YearInvarianceAug(train, self.feature_list.tolist(), size=10000)

        elif self.experiment == 'Experiment5':
            ''' Train invariance between crops for 2016/2017 '''
            train = self.data[self.data.Year != 2018]
            ts_data = CropInvarianceAug(train, self.feature_list.tolist(), size=10000)

        elif self.experiment == 'Experiment6':
            ''' Train invariance between crops for 2016/2017 + 5% from 2018 '''
            train, _ = self.experiment4()
            ts_data = CropInvarianceAug(train, self.feature_list.tolist(), size=10000)

        elif self.experiment == 'Experiment7':
            ''' Train invariance between crops for 2016/2017 + 10% from 2018 '''
            train, _ = self.experiment5()
            ts_data = CropInvarianceAug(train, self.feature_list.tolist(), size=10000)

        elif self.experiment == 'Experiment8':
            ''' Train invariance with all data '''
            train, _ = self.experiment3()
            ts_data = TSAugmented(train, factor=5, feature_list=self.feature_list.tolist())

        elif self.experiment == 'Experiment9':
            ''' Train for 2016/2017 '''
            train = self.data[self.data.Year != 2018]
            ts_data = TSAugmented(train, factor=5, feature_list=self.feature_list.tolist())

        elif self.experiment == 'Experiment10':
            ''' Train for 2016/2017 + 5% from 2018 '''
            train, _ = self.experiment4()
            ts_data = TSAugmented(train, factor=5, feature_list=self.feature_list.tolist())

        elif self.experiment == 'Experiment11':
            ''' Train for 2016/2017 + 10% from 2018 '''
            train, _ = self.experiment5()
            ts_data = TSAugmented(train, factor=5, feature_list=self.feature_list.tolist())

        else:
            print('Experiment not definend')

        
       # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            self.train = ts_data
            self.validate = ts_data
            print(len(self.train))


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)






