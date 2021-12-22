import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, AddNoise
import pandas as pd
import random
import numpy as np
import pandas as pd
import geopandas as gpd

def clean_bavarian_labels(dataframe):
    df = dataframe.copy()

    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df['Year'] = df.Date.dt.year
    df.Date = df.Date.dt.strftime('%m-%d')

    if df.NC.dtypes == 'int':

        df.loc[(df.NC == 600)|(df.NC == 601)|(df.NC == 602), 'NC'] = 1  #Kartoffeln jetzt alle Code 600
        df.loc[(df.NC == 131)|(df.NC == 476), 'NC'] = 2  #Wintergerste jetzt alle Code 131
        df.loc[(df.NC == 400)|(df.NC == 411)|(df.NC == 171)|(df.NC == 410)|(df.NC == 177), 'NC'] = 3  #Mais jetzt alle Code 400
        df.loc[(df.NC == 311)|(df.NC == 489), 'NC'] = 4  #Winterraps jetzt alle Code 311
        df.loc[(df.NC == 115), 'NC'] = 5  #WW
        df.loc[(df.NC == 603), 'NC'] = 6  #ZR

        df.loc[~((df.NC == 1)|(df.NC == 2)|(df.NC == 3)|(df.NC == 4)|(df.NC == 5)|(df.NC == 6)), 'NC'] = 0   #rejection class other

    else:
        print('Label type not int')
    
    return df

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

        #get augmentation of x based on different years
        field = self.xy[self.xy.id == field_idx]
        #print(field.head())
        croptype = field['NC'].values[0]
        year = field["Year"].values[0]
        #print("year",year,type(year))
        yearlist = [2016,2017,2018]
        year1,year2 = self.get_other_years(year, yearlist)
        
        augment1 = self.xy[(self.xy.NC == croptype)&(self.xy['Year'] == year1)]
        augment2 = self.xy[(self.xy.NC == croptype)&(self.xy['Year'] == year2)]

        valid=True
        n=10
        while valid:
            try:
                _id = random.choice(list(set(augment1.id.values)))
                _id2 = random.choice(list(set(augment2.id.values)))

                if isinstance(int(_id), int) and isinstance(int(_id2), int):
                    valid=False
            except:
                print('Exception in random id selection:')
                if n == 0:
                    break
                else:
                    n -= 1


        x1 = augment1[augment1.id == int(_id)][self.feature_list].values
        x2 = augment2[augment2.id == int(_id2)][self.feature_list].values

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

import datetime
#load data for bavaria
bavaria_train = pd.read_excel(
    "/workspace/WorldCrops/data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx")
#bavaria_test = pd.read_excel(
 #   "/workspace/WorldCrops/data/cropdata/Bavaria/sentinel-2/Test_bavaria.xlsx")

train = clean_bavarian_labels(bavaria_train)
#test = clean_bavarian_labels(bavaria_test)

feature_list = train.columns[train.columns.str.contains('B')]
ts_dataset = TimeSeriesPhysical(train, feature_list.tolist(), 'NC')
#ts_dataset_test = TimeSeriesPhysical(test, feature_list.tolist(), 'NC')

batch_size=16
for i in range(1000):
    ts_dataset[i]