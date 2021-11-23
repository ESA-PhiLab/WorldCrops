
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

class OwnAugmentation():

  def jitter(x, sigma=0.03):
      # https://arxiv.org/pdf/1706.00527.pdf
      return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

  def scaling(x, sigma=0.1):
      # https://arxiv.org/pdf/1706.00527.pdf
      factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[1]))
      return np.multiply(x, factor[:,:])


from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, AddNoise
my_augmenter = (
     #TimeWarp() * 5  # random time warping 5 times in parallel
     #+ Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
     #Drift(max_drift=(0.1, 0.5)) @ 0.8  # with 80% probability, random drift the signal up to 10% - 50%
     AddNoise(scale=0.1)
     #+ Reverse() @ 0.5  # with 50% probability, reverse the sequence 
)


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