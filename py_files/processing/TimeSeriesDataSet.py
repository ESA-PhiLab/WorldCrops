from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class OwnAugmentation():

    def jitter(x, sigma=0.03):
        # https://arxiv.org/pdf/1706.00527.pdf
        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

    def scaling(x, sigma=0.1):
        # https://arxiv.org/pdf/1706.00527.pdf
        factor = np.random.normal(
            loc=1., scale=sigma, size=(x.shape[0], x.shape[1]))
        return np.multiply(x, factor[:, :])


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

        # augmentation with jitter and scaling
        aug_x1 = self.x2torch(OwnAugmentation.jitter(x))
        aug_x2 = self.x2torch(OwnAugmentation.scaling(x))

        return (aug_x1, aug_x2), torchx, torchy
        # return torchx, torchy

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
