# %%
# compare the crop type classification of RF and SimSiam
import sys
import os
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

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import copy
#tsai could be helpful
#from tsai.all import *
#computer_setup()

#some definitions for Transformers
batch_size = 1349

test_size = 0.25
SEED = 42
num_workers=4
shuffle_dataset =True
_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0016612
input_dim = 9
#PositonalEncoding
PA = False

logger1 = TensorBoardLogger("Results/Experimente", name="supervised2")
logger2 = TensorBoardLogger("Results/Experimente", name="supervised2")
logger3 = TensorBoardLogger("Results/Experimente", name="supervised2")
logger4 = TensorBoardLogger("Results/Experimente", name="supervised2")

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

#test function
def test_epoch(model, criterion, test_dl, device):
    model.eval()
    with torch.no_grad():
        losses = list()
        y_true_list = list()
        y_pred_list = list()
        y_score_list = list()

        with tqdm.tqdm(enumerate(test_dl), total=len(test_dl), leave=True) as iterator:
            for idx, batch in iterator:
                x, y_true = batch
                logprobabilities = model.forward(x.to(device))
                loss = criterion(logprobabilities, y_true.to(device))
                iterator.set_description(f"test loss={loss:.2f}")
                losses.append(loss)
                y_true_list.append(y_true)
                #print(logprobabilities.shape)
                y_pred_list.append(logprobabilities.argmax(-1))
                y_score_list.append(logprobabilities.exp())

        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list), torch.cat(y_score_list)

# %%
#load data for bavaria
bavaria_reordered = pd.read_excel(
    '../../data/cropdata/Bavaria/sentinel-2/data2016-2018.xlsx', index_col=0)
bavaria_test_reordered = pd.read_excel(
    '../../data/cropdata/Bavaria/sentinel-2/TestData.xlsx', index_col=0)

#delete first two entries with no change
#bavaria_train = bavaria_train.loc[~((bavaria_train.id == 0)|(bavaria_train.id == 1))]
#bavaria_reordered = bavaria_reordered.loc[~((bavaria_reordered.index == 0)|(bavaria_reordered.index == 1))]

train_RF = utils.clean_bavarian_labels(bavaria_reordered)
train_RF = utils.remove_false_observation_RF(train_RF)
test_RF = utils.clean_bavarian_labels(bavaria_test_reordered)
test_RF = utils.remove_false_observation_RF(test_RF)


#experiment with train/test split for all data
dm_bavaria = BavariaDataModule(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size, num_workers = num_workers, experiment='Experiment1')
#experiment with 16/17 train and 2018 test
dm_bavaria2 = BavariaDataModule(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size, num_workers = num_workers, experiment='Experiment2')
#experiment with 16/17 + 5% 2018 train and 2018 test
dm_bavaria3 = BavariaDataModule(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size, num_workers = num_workers, experiment='Experiment3')
#experiment with 16/17 + 10% 2018 train and 2018 test
dm_bavaria4 = BavariaDataModule(data_dir = '../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx', batch_size = batch_size, num_workers = num_workers, experiment='Experiment4')

# %%
# Plots the optimal learning rate

model = Attention_LM(input_dim=input_dim, num_classes = 6, n_head=4, nlayers=3, batch_size = batch_size, lr=lr, seed=42, PositonalEncoding=PA)
model_copy = copy.deepcopy(model)
#torch.save(model_copy, "../model/pretrained/orginal_model1.ckpt")

# %%
#trainer = pl.Trainer(auto_lr_find=True)
#lr_finder = trainer.tuner.lr_find(model, datamodule = dm_bavaria)#, min_lr=0.001, max_lr=0.005, mode='linear')

#fig = lr_finder.plot(suggest=True)
#fig.show()
#print(lr_finder.suggestion())

# find batch size
#trainer = pl.Trainer(deterministic=True, max_epochs=50, check_val_every_n_epoch=10, auto_scale_batch_size='binsearch')
#optimal_batch_size = trainer.tune(model, datamodule = dm_bavaria)
#print(f"Found best batch size to be: {optimal_batch_size}")
#fig.savefig('lr.png')


# %%
############################################################################
# Random Forest & Transformer
############################################################################
# 
#seed_everything(42, workers=True)
trainer = pl.Trainer( deterministic=True, max_epochs= _epochs, logger=logger1)
trainer.fit(model, datamodule=dm_bavaria)
trainer.test(model, datamodule=dm_bavaria)

# %%
# Second experiment train 2016 and 17 - test on 2018
model2 = Attention_LM(input_dim=input_dim, num_classes = 6, n_head=4, nlayers=3, batch_size = batch_size, lr=lr, PositonalEncoding=PA)
model_copy2 = copy.deepcopy(model2)
#torch.save(model_copy2, "../model/pretrained/orginal_model2.ckpt")
trainer = pl.Trainer( deterministic=True, max_epochs= _epochs, logger=logger2)

trainer.fit(model2, datamodule = dm_bavaria2)
trainer.test(model2, datamodule = dm_bavaria2)

# %%
trainer = pl.Trainer( deterministic=True, max_epochs= _epochs, logger=logger3)
model3 = Attention_LM(input_dim=input_dim, num_classes = 6, n_head=4, nlayers=3, batch_size = batch_size, lr=lr, PositonalEncoding=PA)
model_copy3 = copy.deepcopy(model3)
#torch.save(model_copy3, "../model/pretrained/orginal_model3.ckpt")

trainer.fit(model3, datamodule = dm_bavaria3)
trainer.test(model3, datamodule = dm_bavaria3)

# %%
trainer = pl.Trainer( deterministic=True, max_epochs= _epochs, logger=logger4)
model4 = Attention_LM(input_dim=input_dim, num_classes = 6, n_head=4, nlayers=3, batch_size = batch_size, lr=lr, PositonalEncoding=PA)
model_copy4 = copy.deepcopy(model4)
#torch.save(model_copy3, "../model/pretrained/orginal_model3.ckpt")

trainer.fit(model4, datamodule = dm_bavaria4)
trainer.test(model4, datamodule = dm_bavaria4)
# %%


# %%

 # %%
#test without lightning
#losses, y_true, y_pred, y_score = test_epoch( model, torch.nn.CrossEntropyLoss(), dm_bavaria.test_dataloader(), device )

#print("1. Experiment:")
#print(classification_report(y_true.cpu(), y_pred.cpu()))
#print("OA:",round(accuracy_score(y_true, y_pred),2))

#losses2, y_true2, y_pred2, y_score2 = test_epoch( model2, torch.nn.CrossEntropyLoss(), dm_bavaria2.test_dataloader(), device )
#losses3, y_true3, y_pred3, y_score3 = test_epoch( model3, torch.nn.CrossEntropyLoss(), dm_bavaria3.test_dataloader(), device )
#losses4, y_true4, y_pred4, y_score4 = test_epoch( model4, torch.nn.CrossEntropyLoss(), dm_bavaria4.test_dataloader(), device )

#print("2. Experiment:")
#print(classification_report(y_true2.cpu(), y_pred2.cpu()))
#print("OA:",round(accuracy_score(y_true2, y_pred2),2))

#print("3. Experiment:")
#print(classification_report(y_true3.cpu(), y_pred3.cpu()))
#print("OA:",round(accuracy_score(y_true3, y_pred3),2))

#print("4. Experiment:")
#print(classification_report(y_true4.cpu(), y_pred4.cpu()))
#print("OA:",round(accuracy_score(y_true4, y_pred4),2))


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
# %%
#utils.printConfusionResults(confusion)

from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_test =confusion['y_test'].values
y_pred = confusion['y_pred'].values
classification_report(y_test,y_pred)
# %%
print("Overall Accuracy:", accuracy_score(y_test,y_pred))

# %%

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Blues)
#disp.plot()

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
train = train_RF[train_RF.Year != 2018].copy()
X_train = train[train.columns[train.columns.str.contains(band2, na=False)]]
y_train = train['NC']

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

confusion = pd.DataFrame()
confusion['y_pred'] = y_pred
confusion['y_test'] = y_test.values
#printConfusionResults(confusion)



# %%
############################################################################
# Random Forest - 2018 5% for test
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

#sample some examples from 2018
_2018 = train_RF[(train_RF.Year == 2018)].copy()
samples = pd.DataFrame()
percent = 5

for j in range(percent):
    for i in range(6):
        sample = _2018[(_2018.NC == i)].sample(1)
        samples = pd.concat([samples,sample],axis=0)
        _2018.drop(sample.index,inplace=True)

#######

train_1617 = train_RF[train_RF.Year != 2018].copy()
train  = pd.concat([train_1617,samples],axis = 0)
X_train = train[train.columns[train.columns.str.contains(band2, na=False)]]
y_train = train['NC']

test_RF = _2018.copy()
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

confusion = pd.DataFrame()
confusion['y_pred'] = y_pred
confusion['y_test'] = y_test.values

# %%
############################################################################
# Random Forest - 2018 10% for test
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

#sample some examples from 2018
_2018 = train_RF[(train_RF.Year == 2018)].copy()
samples = pd.DataFrame()
percent = 10

for j in range(percent):
    for i in range(6):
        sample = _2018[(_2018.NC == i)].sample(1)
        samples = pd.concat([samples,sample],axis=0)
        _2018 = _2018.drop(sample.index)

#######

train_1617 = train_RF[train_RF.Year != 2018].copy()
train  = pd.concat([train_1617,samples],axis = 0)
X_train = train[train.columns[train.columns.str.contains(band2, na=False)]]
y_train = train['NC']

test_RF = _2018.copy()
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

confusion = pd.DataFrame()
confusion['y_pred'] = y_pred
confusion['y_test'] = y_test.values

# %%
