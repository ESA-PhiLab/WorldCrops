from datetime import datetime
import random
import pandas as pd
import sklearn
import os
import torch
import numpy as np


def rewrite_id_CustomDataSet(df):
    #rewrite the 'id' in order to have an ascending order
    newid = 0
    df = df.copy()
    groups = df.groupby('id')
    for id, group in groups:
        df.loc[df.id == id, 'id'] = newid
        newid +=1
    return df

def clean_bavarian_labels(dataframe):
    df = dataframe.copy()

    if 'Unnamed: 0' in df.columns:
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
    if 'Date' in df.columns:
        df['Year'] = df.Date.dt.year
        df.Date = df.Date.dt.strftime('%m-%d')
        
    df.NC = df.NC.astype(str).astype(int)

    df.loc[(df.NC == 601)|(df.NC == 602), 'NC'] = 600  #Potato
    df.loc[(df.NC == 131)|(df.NC == 476), 'NC'] = 131  #Winter barley
    df.loc[(df.NC == 411)|(df.NC == 171)|(df.NC == 410)|(df.NC == 177), 'NC'] = 400  #Corn
    df.loc[(df.NC == 311)|(df.NC == 489), 'NC'] = 311  #Winter rapeseed
    #WW = 115
    #ZR = 603
    df.loc[~((df.NC == 600)|(df.NC == 131)|(df.NC == 400)|(df.NC == 311)|(df.NC == 115)|(df.NC == 603)), 'NC'] = 6  #rejection class Other

    #rewrite classes for easy handling
    df.loc[(df.NC == 600), 'NC'] = 0 
    df.loc[(df.NC == 131), 'NC'] = 1 
    df.loc[(df.NC == 400), 'NC'] = 2 
    df.loc[(df.NC == 311), 'NC'] = 3 
    df.loc[(df.NC == 115), 'NC'] = 4 
    df.loc[(df.NC == 603), 'NC'] = 5

    #delete class 6 which is the class Other with various unidentified crops
    df = df[df.NC != 6]

    return df

def get_other_years(currentyear, yearlist):
    if currentyear not in yearlist:
        print('Year not in list')
        return currentyear
    else:
        yearlist = yearlist.copy()

        if len(yearlist) > 2:
            yearlist.remove(currentyear)
            output = random.sample(yearlist, len(yearlist))
        else:
            output = random.sample(yearlist, len(yearlist))

        return output[0],output[1]

def augment_df(_df, years):
    ''' 
    dataframe: _df needs 'id', 'NC' and 'Year' column
    years: amount of years for augmentation
    '''

    if not ('NC' in _df.columns) & ('id' in _df.columns) & ('Year' in _df.columns):
        print('Verify the necessary columns NC, id, year')
        return

    train = _df.copy()
    groups = train.groupby('id')
    for id, group in groups:
        croptype = group['NC'].values[0]
        year = group['Year'].values[0]
        year1,year2 = get_other_years(year, years)

        try:
            #filter by potential fields ..same crop type/ different years
            augment1 = train[(train.NC == croptype)&(train['Year'] == year1)]
            augment2 = train[(train.NC == croptype)&(train['Year'] == year2)]

            #in case no data for other than current year is available
            if len(augment1) == 0:
                augment1 = train[(train.NC == croptype)&(train['Year'] == year)]
                augment2 = train[(train.NC == croptype)&(train['Year'] == year)]

            #get random id 
            _id = random.choice(list(set(augment1.id.values)))
            _id2 = random.choice(list(set(augment2.id.values)))

            x1 = augment1[augment1.id == int(_id)]
            x2 = augment2[augment2.id == int(_id2)]

            train.loc[train.id == id, 'id_x1'] = int(x1.head(1).id.values[0])
            train.loc[train.id == id, 'id_x2'] = int(x2.head(1).id.values[0])

        except:
            print('Error in Augmentation')
            pass

    return train

def get_embeddings_plot():
    """Creates a scatter plot with image overlays.
    """
    # initialize empty figure and add subplot
    fig = plt.figure()
    fig.suptitle('Scatter Plot of the Sentinel-2 Dataset')
    ax = fig.add_subplot(1, 1, 1)
    # shuffle images and find out which images to show
    shown_images_idx = []
    shown_images = np.array([[1., 1.]])
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    for i in iterator:
        # only show image if it is sufficiently far away from the others
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 2e-3:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)

    for idx in shown_images_idx:
        circle = plt.Circle((embeddings_2d[idx][0], embeddings_2d[idx][1]), 0.02, color='r')
        ax.add_artist(circle)

    # set aspect ratio
    ratio = 1. / ax.get_data_ratio()
    ax.set_aspect(ratio, adjustable='box')

def printConfusionResults(confusion):
    ''' confusion matrix: 
    Input: dataframe with ['y_pred'] and ['y_test'] columns '''
    
    #PA
    tmp = pd.crosstab(confusion["y_test"],confusion["y_pred"],margins=True,margins_name='Total').T
    tmp['UA']=0
    for idx, row in tmp.iterrows(): 
        #print(idx)
        tmp['UA'].loc[idx] = round(((row[idx])/ row['Total']*100),2)

    #UA
    tmp2 = pd.crosstab(confusion["y_test"],confusion["y_pred"],margins=True,margins_name='Total')
    tmp['PA']=0
    for idx, row in tmp2.iterrows(): 
        #print(row[idx],row.sum())
        tmp['PA'].loc[idx] = round(((row[idx])/ row['Total'])*100,2)


    #hier überprüfen ob alles stimmt

    print('Diag:', tmp.values.diagonal().sum()-tmp['Total'].tail(1)[0] )
    print('Ref:', tmp['Total'].tail(1).values[0])
    oa = (tmp.values.diagonal().sum() - tmp['Total'].tail(1)[0]) / tmp['Total'].tail(1)[0]
    print('OverallAccurcy:',oa)

    print('Kappa:',round(sklearn.metrics.cohen_kappa_score(confusion["y_pred"],confusion["y_test"],weights='quadratic'),4))
    print('#########')
    print("Ac:",round( sklearn.metrics.accuracy_score(confusion["y_pred"],confusion["y_test"]) ,4))
    print(tmp)


def remove_false_observation( df ):
    groups = df.groupby(['id'])
    keys = groups.groups.keys()
    _ids_noise = list()

    for i in keys:
        a = groups.get_group(i)['B9_mean'].to_numpy()
        if (a[0] == a).all(0): 
            _ids_noise.append(groups.get_group(i)['id'].head(1).values[0])

    for id in _ids_noise:
        df = df.drop(df[df.id == id].index)
    return df

def remove_false_observation_RF( df ):
    ids_noise = list()
    for idx, row in df.iterrows():
        a = row[row.index.str.contains('B9')].to_numpy()
        if (a[0] == a).all(0):
            df = df.drop(df.index[idx])
    return df


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True