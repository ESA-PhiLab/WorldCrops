# %%
# compare the crop type classification of RF and SimSiam
import sys
sys.path.append('./model')
sys.path.append('..')

from model import *
from processing import *
import math

import torch.nn as nn
import torchvision
import lightly

import contextily as ctx
import matplotlib.pyplot as plt
import breizhcrops
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
from breizhcrops import BreizhCrops
from breizhcrops.datasets.breizhcrops import BANDS as allbands

import torch
import tqdm

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

import sklearn.datasets
import pandas as pd
import numpy as np
import umap
import umap.plot

#tsai could be helpful
#from tsai.all import *
#computer_setup()
# %%

def printConfusionResults(confusion, logfile='log.xlsx'):
    # PA
    tmp = pd.crosstab(
        confusion["y_test"], confusion["y_pred"], margins=True, margins_name='Total').T
    tmp['UA'] = 0
    for idx, row in tmp.iterrows():
        # print(idx)
        tmp['UA'].loc[idx] = round(((row[idx]) / row['Total']*100), 2)

    # UA
    tmp2 = pd.crosstab(
        confusion["y_test"], confusion["y_pred"], margins=True, margins_name='Total')
    tmp['PA'] = 0
    for idx, row in tmp2.iterrows():
        # print(row[idx],row.sum())
        tmp['PA'].loc[idx] = round(((row[idx]) / row['Total'])*100, 2)

    # hier überprüfen ob alles stimmt
    print('Diag:', tmp.values.diagonal().sum()-tmp['Total'].tail(1)[0])
    print('Ref:', tmp['Total'].tail(1).values[0])
    oa = (tmp.values.diagonal().sum() -
          tmp['Total'].tail(1)[0]) / tmp['Total'].tail(1)[0]
    print('OverallAccurcy:', round(oa, 2))

    print('Kappa:', round(sklearn.metrics.cohen_kappa_score(
        confusion["y_pred"], confusion["y_test"], weights='quadratic'), 2))
    print('#########')
    print("Ac:", round(sklearn.metrics.accuracy_score(
        confusion["y_pred"], confusion["y_test"]), 2))

    # tmp.to_excel("Daten/Neu/"+logfile+".xlsx")
    print(tmp)

# %%
#load data for bavaria
bavaria_train = pd.read_excel(
    "../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx")
bavaria_test = pd.read_excel(
    "../../data/cropdata/Bavaria/sentinel-2/Test_bavaria.xlsx")

bavaria_reordered = pd.read_excel(
    '../../data/cropdata/Bavaria/sentinel-2/data2016-2018.xlsx', index_col=0)
bavaria_test_reordered = pd.read_excel(
    '../../data/cropdata/Bavaria/sentinel-2/TestData.xlsx', index_col=0)

#delete first two entries with no change
#bavaria_train = bavaria_train.loc[~((bavaria_train.id == 0)|(bavaria_train.id == 1))]
#bavaria_reordered = bavaria_reordered.loc[~((bavaria_reordered.index == 0)|(bavaria_reordered.index == 1))]

#bavaria_train.to_excel (r'test.xlsx', index = False, header=True)
train = utils.clean_bavarian_labels(bavaria_train)
test = utils.clean_bavarian_labels(bavaria_test)

train_RF = utils.clean_bavarian_labels(bavaria_reordered)
test_RF = utils.clean_bavarian_labels(bavaria_test_reordered)

#delete class 0
train = train[train.NC != 0]
test = test[test.NC != 0]

train_RF = train_RF[train_RF.NC != 0]
test_RF = test_RF[test_RF.NC != 0]

#rewrite the 'id' as we deleted one class
#only for bavaria_test and bavaria_train
newid = 0
groups = train.groupby('id')
for id, group in groups:
    train.loc[train.id == id, 'id'] = newid
    newid +=1

test = test[test.NC != 0]
#rewrite the 'id' as we deleted one class
newid = 0
groups = test.groupby('id')
for id, group in groups:
    test.loc[test.id == id, 'id'] = newid
    newid +=1

# %%
train.head()
# %%

# %%
_model = Attention_LM()
_model
# %%



 # %%
# %%
############################################################################
# Random Forest & Transformer
############################################################################
# Custom DataSet with augmentation
# augmentation needs to be extended

feature_list = train.columns[train.columns.str.contains('B')]
ts_dataset = TimeSeriesPhysical(train, feature_list.tolist(), 'NC')
ts_dataset_test = TimeSeriesDataSet(test, feature_list.tolist(), 'NC')

batch_size=32
dataloader_train = torch.utils.data.DataLoader(
    ts_dataset, batch_size=batch_size, shuffle=True,drop_last=False, num_workers=2)
dataloader_test = torch.utils.data.DataLoader(
    ts_dataset_test, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=2)

# %% 


# %%      
############################################################################
# Random Forest all data
############################################################################

# hier parameter festlegen
# RF:
_n_estimators = 1000
_max_features = 'auto'
_J = 0
_test_size = 0.25
_cv = KFold(n_splits=5, shuffle=True, random_state=_J)

# without other and for bands
band2 = "_B"
#_wO = bavaria_reordered[bavaria_reordered.NC != 1]
X = train_RF[train_RF.columns[train_RF.columns.str.contains(band2, na=False)]]
y = train_RF['NC']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=_test_size, random_state=_J)

clf = RandomForestClassifier(
    n_estimators=_n_estimators, max_features=_max_features, random_state=_J)
clf.fit(X_train, y_train)
y_pred = clf.predict(X=X_test)
proba = clf.predict_proba(X_test)

print('###########Band and RF ##############')
print('Accuracy of classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))
score = cross_val_score(clf, X, y, cv=_cv)
print('Accuracy of classifier Cross Validation: {:.2f}'
      .format(score.mean()))
confusion = pd.DataFrame()
confusion['y_pred'] = y_pred
confusion['y_test'] = y_test.values
#printConfusionResults(confusion)
# %%
printConfusionResults(confusion)
# %%
############################################################################
# Random Forest - 2018 for test
############################################################################

# hier parameter festlegen
# RF:
_n_estimators = 1000
_max_features = 'auto'
_J = 0
_test_size = 0.25
_cv = KFold(n_splits=5, shuffle=True, random_state=_J)

# without other and for bands
band2 = "_B"
#_wO = bavaria_reordered[bavaria_reordered.NC != 1]

test_RF = train_RF[train_RF.Year == 2018].copy()
train_RF = train_RF[train_RF.Year != 2018]
X_train = train_RF[train_RF.columns[train_RF.columns.str.contains(band2, na=False)]]
y_train = train_RF['NC']

X_test = test_RF[test_RF.columns[test_RF.columns.str.contains(band2, na=False)]]
y_test = test_RF['NC']

clf_18 = RandomForestClassifier(
    n_estimators=_n_estimators, max_features=_max_features, random_state=_J)
clf_18.fit(X_train, y_train)
y_pred = clf_18.predict(X=X_test)
proba = clf_18.predict_proba(X_test)

print('###########Band and RF ##############')
print('Accuracy of classifier on training set: {:.2f}'
      .format(clf_18.score(X_train, y_train)))
print('Accuracy of classifier on test set: {:.2f}'
      .format(clf_18.score(X_test, y_test)))

      .format(score.mean()))
confusion = pd.DataFrame()
confusion['y_pred'] = y_pred
confusion['y_test'] = y_test.values
#printConfusionResults(confusion)
# %%
printConfusionResults(confusion)
# %%
############################################################################
# Random Forest - 2018 for test
############################################################################

# hier parameter festlegen
# RF:
_n_estimators = 1000
_max_features = 'auto'
_J = 0
_test_size = 0.25
_cv = KFold(n_splits=5, shuffle=True, random_state=_J)

# without other and for bands
band2 = "_B"
#_wO = bavaria_reordered[bavaria_reordered.NC != 1]

X = train_RF[train_RF.columns[train_RF.columns.str.contains(band2, na=False)]]
y = train_RF['NC']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=_test_size, random_state=_J)

clf_18 = RandomForestClassifier(
    n_estimators=_n_estimators, max_features=_max_features, random_state=_J)
clf_18.fit(X_train, y_train)
y_pred = clf_18.predict(X=X_test)
proba = clf_18.predict_proba(X_test)

print('###########Band and RF ##############')
print('Accuracy of classifier on training set: {:.2f}'
      .format(clf_18.score(X_train, y_train)))
print('Accuracy of classifier on test set: {:.2f}'
      .format(clf_18.score(X_test, y_test)))

      .format(score.mean()))
confusion = pd.DataFrame()
confusion['y_pred'] = y_pred
confusion['y_test'] = y_test.values
#printConfusionResults(confusion)# %%

# %%
train_RF[train_RF.Year == 2018]
# %%
