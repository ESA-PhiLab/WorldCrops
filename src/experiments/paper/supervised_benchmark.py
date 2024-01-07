# %%
"""
    Supervised benchmark results (RF, Transformer) for 
    the crop type data in the publication https://arxiv.org/abs/2204.02100.
"""
import copy
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import tqdm
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import KFold, cross_val_score, train_test_split

#import selfsupervised
from selfsupervised.data.croptypes.datamodules import BavariaDataModule
from selfsupervised.model.lightning.transformer_encoder import \
    TransformerEncoder
from selfsupervised.processing.utils import (clean_bavarian_labels,
                                             remove_false_observation_RF,
                                             seed_torch)

# %%
################################################################
# Configuration
################################################################

if os.path.isfile(sys.argv[1]) and os.access(sys.argv[1], os.R_OK):
    # Open both config files as dicts and combine them into a single dict.
    with open(sys.argv[1], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
else:
    with open("../../../config/croptypes/param_supervised_benchmark.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


#training parameter
batch_size = cfg["training"]['batch_size']
num_workers = cfg["training"]['num_workers']
test_size = cfg["training"]['test_size']
SEED = cfg["training"]['seed']
shuffle_dataset = cfg["training"]['shuffle_dataset']
epochs = cfg["training"]['epochs']
device = cfg["training"]['device']
lr = cfg["training"]['learning_rate']
#positional encoding
pe = cfg["training"]['pe']

#transformer
input_dim = cfg["transformer"]['input_dim']
num_classes = cfg["transformer"]['num_classes']
n_head = cfg["transformer"]['n_head']
nlayers = cfg["transformer"]['nlayers']


#PATH to data files
training_data = '../../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx'

logger1 = TensorBoardLogger("Results/Experimente", name="supervised2")
logger2 = TensorBoardLogger("Results/Experimente", name="supervised2")
logger3 = TensorBoardLogger("Results/Experimente", name="supervised2")
logger4 = TensorBoardLogger("Results/Experimente", name="supervised2")

# %%

seed_torch()

# test function
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
                # print(logprobabilities.shape)
                y_pred_list.append(logprobabilities.argmax(-1))
                y_score_list.append(logprobabilities.exp())

        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list), torch.cat(y_score_list)


# %%
# load data for Random forest
bavaria_reordered = pd.read_excel(
    '../../../data/cropdata/Bavaria/sentinel-2/data2016-2018.xlsx', index_col=0)
bavaria_test_reordered = pd.read_excel(
    '../../../data/cropdata/Bavaria/sentinel-2/TestData.xlsx', index_col=0)

train_RF = clean_bavarian_labels(bavaria_reordered)
train_RF = remove_false_observation_RF(train_RF)
test_RF = clean_bavarian_labels(bavaria_test_reordered)
test_RF = remove_false_observation_RF(test_RF)


#load data for our experiments
# experiment with train/test split for all data
dm_bavaria = BavariaDataModule(data_dir=training_data,
                               batch_size=batch_size, num_workers=num_workers, experiment='Experiment1')
# experiment with 16/17 train and 2018 test
dm_bavaria2 = BavariaDataModule(data_dir=training_data,
                                batch_size=batch_size, num_workers=num_workers, experiment='Experiment2')
# experiment with 16/17 + 5% 2018 train and 2018 test
dm_bavaria3 = BavariaDataModule(data_dir=training_data,
                                batch_size=batch_size, num_workers=num_workers, experiment='Experiment3')
# experiment with 16/17 + 10% 2018 train and 2018 test
dm_bavaria4 = BavariaDataModule(data_dir=training_data,
                                batch_size=batch_size, num_workers=num_workers, experiment='Experiment4')

# %%
# Plots the optimal learning rate

model = TransformerEncoder(input_dim=input_dim, num_classes=6, n_head=4,
                             nlayers=3, batch_size=batch_size, lr=lr, seed=42, PositonalEncoding=pe)
model_copy = copy.deepcopy(model)
# torch.save(model_copy, "../model/pretrained/orginal_model1.ckpt")

# %%
# trainer = pl.Trainer(auto_lr_find=True)
# lr_finder = trainer.tuner.lr_find(model, datamodule = dm_bavaria)#, min_lr=0.001, max_lr=0.005, mode='linear')

# fig = lr_finder.plot(suggest=True)
# fig.show()
# print(lr_finder.suggestion())

# find batch size
# trainer = pl.Trainer(deterministic=True, max_epochs=50, check_val_every_n_epoch=10, auto_scale_batch_size='binsearch')
# optimal_batch_size = trainer.tune(model, datamodule = dm_bavaria)
# print(f"Found best batch size to be: {optimal_batch_size}")
# fig.savefig('lr.png')


# %%
############################################################################
# Random Forest & Transformer
############################################################################
#
# seed_everything(42, workers=True)
trainer = pl.Trainer(deterministic=True, max_epochs=epochs, logger=logger1)
trainer.fit(model, datamodule=dm_bavaria)
trainer.test(model, datamodule=dm_bavaria)

# %%
# Second experiment train 2016 and 17 - test on 2018
model2 = TransformerEncoder(input_dim=input_dim, num_classes=6, n_head=4,
                      nlayers=3, batch_size=batch_size, lr=lr, PositonalEncoding=pe)
model_copy2 = copy.deepcopy(model2)
# torch.save(model_copy2, "../model/pretrained/orginal_model2.ckpt")
trainer = pl.Trainer(deterministic=True, max_epochs=epochs, logger=logger2)

trainer.fit(model2, datamodule=dm_bavaria2)
trainer.test(model2, datamodule=dm_bavaria2)

# %%
trainer = pl.Trainer(deterministic=True, max_epochs=epochs, logger=logger3)
model3 = TransformerEncoder(input_dim=input_dim, num_classes=6, n_head=4,
                      nlayers=3, batch_size=batch_size, lr=lr, PositonalEncoding=pe)
model_copy3 = copy.deepcopy(model3)
# torch.save(model_copy3, "../model/pretrained/orginal_model3.ckpt")

trainer.fit(model3, datamodule=dm_bavaria3)
trainer.test(model3, datamodule=dm_bavaria3)

# %%
trainer = pl.Trainer(deterministic=True, max_epochs=epochs, logger=logger4)
model4 = TransformerEncoder(input_dim=input_dim, num_classes=6, n_head=4,
                      nlayers=3, batch_size=batch_size, lr=lr, PositonalEncoding=pe)
model_copy4 = copy.deepcopy(model4)
# torch.save(model_copy3, "../model/pretrained/orginal_model3.ckpt")

trainer.fit(model4, datamodule=dm_bavaria4)
trainer.test(model4, datamodule=dm_bavaria4)
# %%


# %%
# test without lightning
# losses, y_true, y_pred, y_score = test_epoch( model, torch.nn.CrossEntropyLoss(), dm_bavaria.test_dataloader(), device )

# print("1. Experiment:")
# print(classification_report(y_true.cpu(), y_pred.cpu()))
# print("OA:",round(accuracy_score(y_true, y_pred),2))

# losses2, y_true2, y_pred2, y_score2 = test_epoch( model2, torch.nn.CrossEntropyLoss(), dm_bavaria2.test_dataloader(), device )
# losses3, y_true3, y_pred3, y_score3 = test_epoch( model3, torch.nn.CrossEntropyLoss(), dm_bavaria3.test_dataloader(), device )
# losses4, y_true4, y_pred4, y_score4 = test_epoch( model4, torch.nn.CrossEntropyLoss(), dm_bavaria4.test_dataloader(), device )

# print("2. Experiment:")
# print(classification_report(y_true2.cpu(), y_pred2.cpu()))
# print("OA:",round(accuracy_score(y_true2, y_pred2),2))

# print("3. Experiment:")
# print(classification_report(y_true3.cpu(), y_pred3.cpu()))
# print("OA:",round(accuracy_score(y_true3, y_pred3),2))

# print("4. Experiment:")
# print(classification_report(y_true4.cpu(), y_pred4.cpu()))
# print("OA:",round(accuracy_score(y_true4, y_pred4),2))


# %%

# %%
############################################################################
# Random Forest all data
############################################################################

# hier parameter festlegen
# RF:
_n_estimators = 1000
_max_features = 'sqrt'
_J = 0
_test_size = 0.25
_cv = KFold(n_splits=5, shuffle=True, random_state=_J)

# without other and for bands
band2 = "_B"
# _wO = bavaria_reordered[bavaria_reordered.NC != 1]
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
# utils.printConfusionResults(confusion)


y_test = confusion['y_test'].values
y_pred = confusion['y_pred'].values
classification_report(y_test, y_pred)
# %%
print("Overall Accuracy:", accuracy_score(y_test, y_pred))

# %%

disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, cmap=plt.cm.Blues)
# disp.plot()

# %%

############################################################################
# Random Forest - 2018 for test
############################################################################

# hier parameter festlegen
# RF:
_n_estimators = 1000
_max_features = 'sqrt'
_J = 0
_test_size = 0.25
_cv = KFold(n_splits=5, shuffle=True, random_state=_J)

# without other and for bands
band2 = "_B"
# _wO = bavaria_reordered[bavaria_reordered.NC != 1]

test_RF = train_RF[train_RF.Year == 2018].copy()
train = train_RF[train_RF.Year != 2018].copy()
X_train = train[train.columns[train.columns.str.contains(band2, na=False)]]
y_train = train['NC']

X_test = test_RF[test_RF.columns[test_RF.columns.str.contains(
    band2, na=False)]]
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
# printConfusionResults(confusion)


# %%
############################################################################
# Random Forest - 2018 5% for test
############################################################################

# hier parameter festlegen
# RF:
_n_estimators = 1000
_max_features = 'sqrt'
_J = 0
_test_size = 0.25
_cv = KFold(n_splits=5, shuffle=True, random_state=_J)

# without other and for bands
band2 = "_B"
# _wO = bavaria_reordered[bavaria_reordered.NC != 1]

# sample some examples from 2018
_2018 = train_RF[(train_RF.Year == 2018)].copy()
samples = pd.DataFrame()
percent = 5

for j in range(percent):
    for i in range(6):
        sample = _2018[(_2018.NC == i)].sample(1)
        samples = pd.concat([samples, sample], axis=0)
        _2018.drop(sample.index, inplace=True)

#######

train_1617 = train_RF[train_RF.Year != 2018].copy()
train = pd.concat([train_1617, samples], axis=0)
X_train = train[train.columns[train.columns.str.contains(band2, na=False)]]
y_train = train['NC']

test_RF = _2018.copy()
X_test = test_RF[test_RF.columns[test_RF.columns.str.contains(
    band2, na=False)]]
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
_max_features = 'sqrt'
_J = 0
_test_size = 0.25
_cv = KFold(n_splits=5, shuffle=True, random_state=_J)

# without other and for bands
band2 = "_B"
# _wO = bavaria_reordered[bavaria_reordered.NC != 1]

# sample some examples from 2018
_2018 = train_RF[(train_RF.Year == 2018)].copy()
samples = pd.DataFrame()
percent = 10

for j in range(percent):
    for i in range(6):
        sample = _2018[(_2018.NC == i)].sample(1)
        samples = pd.concat([samples, sample], axis=0)
        _2018 = _2018.drop(sample.index)

#######

train_1617 = train_RF[train_RF.Year != 2018].copy()
train = pd.concat([train_1617, samples], axis=0)
X_train = train[train.columns[train.columns.str.contains(band2, na=False)]]
y_train = train['NC']

test_RF = _2018.copy()
X_test = test_RF[test_RF.columns[test_RF.columns.str.contains(
    band2, na=False)]]
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
