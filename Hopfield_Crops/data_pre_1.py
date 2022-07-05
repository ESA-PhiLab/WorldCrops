#%%
import numpy as np 
import pandas as pd

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

def remove_false_observation(df ):
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

def remove_false_observation_RF(df ):
    ids_noise = list()
    for idx, row in df.iterrows():
        a = row[row.index.str.contains('B9')].to_numpy()
        if (a[0] == a).all(0):
            df = df.drop(df.index[idx])
    return df

data_dir = './Bavaria/sentinel-2/Training_bavaria.xlsx'
data = pd.read_excel(data_dir)
#list with selected features "Bands 1-13"
feature_list = data.columns[data.columns.str.contains('B')].tolist()

ddata = clean_bavarian_labels(data)
# data = remove_false_observation(data)

#%%
d = np.array(ddata)
# print(d.shape)
# print(d[0:15])
# print(d[1])
# print(d[2])
data = np.zeros((3, 6, 100, 14, 13))
target = np.zeros((3, 6, 100, 6))
y1=0
y2=0
y3=0
y1t = np.zeros(6)
y2t = np.zeros(6)
y3t = np.zeros(6)

for n in range(0,d.shape[0], 14):
    if str(d[n,-1])=='2016':
        data[0, d[n, 3], int(y1t[d[n, 3]])] = d[n:(n+14), 4:17]
        y1t[d[n, 3]]+=1
        y1+=1
    if str(d[n,-1])=='2017':
        data[1, d[n, 3], int(y2t[d[n, 3]])] = d[n:(n+14), 4:17]
        y2t[d[n, 3]]+=1
        y2+=1
    if str(d[n,-1])=='2018':
        data[2, d[n, 3], int(y3t[d[n, 3]])] = d[n:(n+14), 4:17]
        y3t[d[n, 3]]+=1
        y3+=1

data = np.einsum('abcde->abced', data)

target[:,0,:,0] = 1
target[:,1,:,1] = 1
target[:,2,:,2] = 1
target[:,3,:,3] = 1
target[:,4,:,4] = 1
target[:,5,:,5] = 1

np.save('Bavaria_13', {"data": data, "target": target})
#%%
data = np.load('Bavaria_13.npy', allow_pickle=True).item()
# print(data[0])
print(data['target'].shape)
