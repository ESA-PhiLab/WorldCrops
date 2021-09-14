# %%
# compare the crop type classification of RF and SimSiam

from TimeSeriesDataSet import *
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


# load training and test data for bavaria
bavaria_train = pd.read_excel(
    "../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx")
bavaria_test = pd.read_excel(
    "../data/cropdata/Bavaria/sentinel-2/Test_bavaria.xlsx")
bavaria_test.drop(['Unnamed: 0'], axis=1, inplace=True)
bavaria_train.drop(['Unnamed: 0'], axis=1, inplace=True)

# %%
bavaria_train.tail()
# %%
bavaria_reordered = pd.read_excel(
    '../data/cropdata/Bavaria/sentinel-2/data2016-2018.xlsx', index_col=0)
bavaria_test_reordered = pd.read_excel(
    '../data/cropdata/Bavaria/sentinel-2/TestData.xlsx', index_col=0)

# %%
bavaria_test_reordered.head()
# %%
############################################################################
# Random Forest
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
_wO = bavaria_reordered[bavaria_reordered.NC != 1]
X = _wO[_wO.columns[_wO.columns.str.contains(band2, na=False)]]
y = _wO['NC']

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
printConfusionResults(confusion)
# %%

############################################################################
# SimSiam
############################################################################
# Custom DataSet with augmentation
# augmentation needs to be extended

feature_list = bavaria_train.columns[bavaria_train.columns.str.contains('B')]
dataset = TimeSeriesDataSet(bavaria_train, feature_list.tolist(), 'NC')
print(dataset[100])
augmentation, _, y = dataset[100]
# %%

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=2)
# %%


def plot_batch(augmentation, timeseries, labels):
    fig, axs = plt.subplots(3)
    fig.suptitle('Augmentation')

    #x  = timeseries[:,bands_idxs]
    axs[0].plot(timeseries.numpy().reshape(14, 13), "-")
    axs[1].plot(augmentation[0].numpy().reshape(14, 13), "-")
    axs[2].plot(augmentation[1].numpy().reshape(14, 13), "-")

    print("Crop type:", labels)
    plt.show()


for i, data in enumerate(dataloader):
    if (i > 0):
        break
    aug, timeseries, labels = data
    plot_batch(aug, timeseries, labels)
# %%
# init simsiam and training
# https://docs.lightly.ai/tutorials/package/tutorial_simsiam_esa.html
#
