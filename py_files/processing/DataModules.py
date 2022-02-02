import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from typing import Optional
from .utils import *
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from .TimeSeriesDataSet import *

class BavariaDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size = 32, num_workers = 2):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = pd.read_excel(self.data_dir)
        #list with selected features
        self.feature_list = self.data.columns[self.data.columns.str.contains('B')]

        #preprocess
        self.data = clean_bavarian_labels(self.data)

        #data sets
        self.train = None
        self.validate = None
        self.test = None

    def setup(self, stage: Optional[str] = None):

        # use the full data set and make train test split
        splitter = GroupShuffleSplit(test_size=.25, n_splits=2, random_state = 0)
        split = splitter.split(self.data, groups=self.data['id'])
        train_inds, test_inds = next(split)
        train = self.data.iloc[train_inds]
        test = self.data.iloc[test_inds]

        #callback function for dataframe cleaning, interpolation etc.
        def func(df):
            return clean_bavarian_labels(df)

        ts_train = TSDataSet(train, self.feature_list.tolist(), 'NC', field_id = 'id')
        ts_test = TSDataSet(test, self.feature_list.tolist(), 'NC', field_id = 'id')

        # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            self.train = ts_train
            self.validate = ts_test

        # Assign test dataset for use in dataloader(s)
        if stage in (None, "test"):
            self.test = ts_test


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)


class Bavaria1617DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size = 32, num_workers = 2):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = pd.read_excel(self.data_dir)
        #list with selected features
        self.feature_list = self.data.columns[self.data.columns.str.contains('B')]

        #preprocess
        self.data = clean_bavarian_labels(self.data)

        #data sets
        self.train = None
        self.validate = None
        self.test = None

    def setup(self, stage: Optional[str] = None):

        # use the full data set and make train test split
        test_2018 = self.data[self.data.Year == 2018]
        train_1617 = self.data[self.data.Year != 2018]

        #callback function for dataframe cleaning, interpolation etc.
        def func(df):
            return clean_bavarian_labels(df)

        ts_train = TSDataSet(train_1617, self.feature_list.tolist(), 'NC', field_id = 'id')
        ts_test = TSDataSet(test_2018, self.feature_list.tolist(), 'NC', field_id = 'id')

        # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            self.train = ts_train
            self.validate = ts_test

        # Assign test dataset for use in dataloader(s)
        if stage in (None, "test"):
            self.test = ts_test


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)

class Bavaria1percentDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size = 32, num_workers = 2):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = pd.read_excel(self.data_dir)
        #list with selected features
        self.feature_list = self.data.columns[self.data.columns.str.contains('B')]

        #preprocess
        self.data = clean_bavarian_labels(self.data)

        #data sets
        self.train = None
        self.validate = None
        self.test = None

    def setup(self, stage: Optional[str] = None):

        train_1617 = self.data[self.data.Year != 2018]

        # one percent sample data for 2018
        _2018 = self.data[self.data.Year == 2018]
        samples = pd.DataFrame()
        for i in range(0,6):
            id = _2018[(_2018.NC == i)].sample(1).id
            sample = _2018[(_2018.id == id.values[0])]
            #delete row in orginal 2018 data
            _2018 = _2018.drop(_2018[_2018.id == id.values[0]].index)
            samples = pd.concat([samples,sample],axis=0)
        percent1 = pd.concat([train_1617,samples],axis = 0)

        #callback function for dataframe cleaning, interpolation etc.
        def func(df):
            return clean_bavarian_labels(df)

        ts_train = TSDataSet(percent1, self.feature_list.tolist(), 'NC', field_id = 'id')
        ts_test = TSDataSet(_2018, self.feature_list.tolist(), 'NC', field_id = 'id')

        # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            self.train = ts_train
            self.validate = ts_test

        # Assign test dataset for use in dataloader(s)
        if stage in (None, "test"):
            self.test = ts_test


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)
