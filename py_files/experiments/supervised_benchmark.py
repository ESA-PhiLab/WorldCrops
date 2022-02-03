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

from torch.utils.data.sampler import SubsetRandomSampler

#tsai could be helpful
#from tsai.all import *
#computer_setup()

#some definitions for Transformers
batch_size = 32
test_size = 0.25
SEED = 42
num_workers=4
shuffle_dataset =True
_epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
#delete class 6 which is the class Other with various unidentified crops
train = train[train.NC != 6]
test = test[test.NC != 6]

train_RF = utils.clean_bavarian_labels(bavaria_reordered)
test_RF = utils.clean_bavarian_labels(bavaria_test_reordered)
#delete class 6
train_RF = train_RF[train_RF.NC !=6]
test_RF = test_RF[test_RF.NC != 6]

train = utils.rewrite_id_CustomDataSet(train)
test = utils.rewrite_id_CustomDataSet(test)


# %%
train.id.unique()

# %%

#split data to 1617 and 2018 
test_2018 = train[train.Year == 2018].copy()
train_1617 = train[train.Year != 2018]

train_1617 = utils.rewrite_id_CustomDataSet(train_1617)
test_2018 = utils.rewrite_id_CustomDataSet(test_2018)

feature_list = train_1617.columns[train_1617.columns.str.contains('B')]
ts_data_train = TimeSeriesDataSet(train_1617, feature_list.tolist(), 'NC')
ts_data_test = TimeSeriesDataSet(test_2018, feature_list.tolist(), 'NC')

dataloader1617_train = torch.utils.data.DataLoader(ts_data_train, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=num_workers)
dataloader18_validate = torch.utils.data.DataLoader(ts_data_test, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=num_workers)
dataloader18_test = torch.utils.data.DataLoader(ts_data_test, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=num_workers)

# use the full data set and make train test split
from sklearn.model_selection import GroupShuffleSplit
splitter = GroupShuffleSplit(test_size=.25, n_splits=2, random_state = 0)
split = splitter.split(train, groups=train['id'])
train_inds, test_inds = next(split)
train_all = train.iloc[train_inds]
test_all = train.iloc[test_inds]

train_all = utils.rewrite_id_CustomDataSet(train_all)
test_all = utils.rewrite_id_CustomDataSet(test_all)

ts_train_all = TimeSeriesDataSet(train_all, feature_list.tolist(), 'NC')
ts_test_all = TimeSeriesDataSet(test_all, feature_list.tolist(), 'NC')

dataloader_train_all = DataLoader(ts_train_all,batch_size=batch_size, shuffle=True,drop_last=True,num_workers=num_workers)
dataloader_test_all = DataLoader(ts_test_all,batch_size=batch_size, shuffle=True,drop_last=True,num_workers=num_workers)

# one percent sample data for 2018
_2018 = train[train.Year == 2018].copy()
samples = pd.DataFrame()
for i in range(0,6):
    id = _2018[(_2018.NC == i)].sample(1).id
    sample = _2018[(_2018.id == id.values[0])]
    samples = pd.concat([samples,sample],axis=0)

percent1 = pd.concat([train_1617,samples],axis = 0)

percent1 = utils.rewrite_id_CustomDataSet(percent1)

ts_data_train = TimeSeriesDataSet(percent1, feature_list.tolist(), 'NC')
dataloader18_1percent_train = torch.utils.data.DataLoader(ts_data_train, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=num_workers)

# %%
##########################
# Attention Transformer 
##########################

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
class Attention_LM2(pl.LightningModule):

    def __init__(self, input_dim = 13, num_classes = 7, d_model = 64, n_head = 2, d_ffn = 128, nlayers = 2, dropout = 0.018, activation="relu", lr = 0.0002):
        super().__init__()
        """
        Args:
            input_dim: amount of input dimensions -> Sentinel2 has 13 bands
            num_classes: amount of target classes
            dropout: default = 0.018
            d_model: default = 64 #number of expected features
            n_head: default = 2 #number of heads in multiheadattention models
            d_ff: default = 128 #dim of feedforward network 
            nlayers: default = 2 #number of encoder layers
            + : https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        Input:
            batch size(N) x T x D
        Output
            batch size(N) x Targets
        """

        self.model_type = 'Transformer_LM'
        self.losses = list()

        # Hyperparameters
        self.lr = lr
        self.ce = nn.CrossEntropyLoss()
        self.save_hyperparameters()

        # Layers
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout = dropout, activation=activation, batch_first=True)
        self.inlinear = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU()
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers, nn.LayerNorm(d_model))
        self.outlinear = nn.Linear(d_model, num_classes)

        
    def forward(self,x):
        # N x T x D -> N x T x d_model / Batch First!
        x = self.inlinear(x) 
        x = self.relu(x)
        x = self.transformer_encoder(x)
        x = x.max(1)[0]
        x = self.relu(x)
        x = self.outlinear(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        #accuracy = accuracy_score(y,y_pred)
        self.log('test_results', {'test_loss' : loss},prog_bar=True)
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y, 'y_pred_idx': y_pred.argmax(-1), 'y_score': y_pred.exp()}

    def test_step_end(self, output):
        return output

    def test_epoch_end(self, outputs):
        accuracy = list()
        y_true_list = list()
        y_pred_list = list()
        y_score_list = list()

        for item in outputs:
            self.losses.append(item['y_true'])
            y_pred_list.append(item['y_pred'])


        #print(accuracy_score(torch.cat(y_true_list),torch.cat(y_pred_list)))
        self.log('y_true_list',torch.cat(self.losses), on_step=False, on_epoch=True, prog_bar=True, logger=True) 


            #for out in test_outputs:
                #print(out)
                #y_true = out['test_results']['y_true']
                #y_pred = out['test_results']['y_pred']
                #accuracy.append(accuracy_score(y_true,y_pred))

            #y_true_list.append(out['y_true'])
            #y_pred_list.append(logprobabilities.argmax(-1))
            #y_score_list.append(logprobabilities.exp())

        #print(f"Test Accuracy: {round(torch.mean(torch.stack(accuracy)),2)}")



# %%
############################################################################
# Random Forest & Transformer
############################################################################

# First experiment train test split with all data
# Parametrisation based on https://arxiv.org/pdf/1905.11893.pdf
model1 = Attention_LM2(num_classes = 6,n_head=2,nlayers=3)
#model1.train()

trainer = pl.Trainer( gpus=1 if str(device).startswith("cuda") else 0, deterministic=True, max_epochs=_epochs)
trainer.fit(model1, dataloader_train_all)
# %%
bla= trainer.test(model1, dataloader_train_all)
# %%
bla
# %%
test = {'test_results': {'test_loss': 1.1572, 'y_pred': -2.3070,'y_true': 2.5126}}

test['test_results']

# %%
#irgendwas stimmt nicht mit dem dataloader 
#die utils function mit id in timeseries einbauen

# Second experiment train 2016 and 17 - test on 2018
"""model2 = Attention_LM(num_classes = 6, nlayers=3)
model2.train()
trainer = pl.Trainer( gpus=1 if str(device).startswith("cuda") else 0, deterministic=True, max_epochs=_epochs)
trainer.fit(model2, dataloader1617_train)
# Third experiment train 2016 and 17 + 1 percent from 2018 - test on 2018
model3 = Attention_LM(num_classes = 6, nlayers=3)
model3.train()
trainer = pl.Trainer( gpus=1 if str(device).startswith("cuda") else 0, deterministic=True, max_epochs=_epochs)
trainer.fit(model3, dataloader18_1percent_train)"""

# %%
#trainer.test(_model,test_dl)

# %%

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
                print(logprobabilities.shape)
                y_pred_list.append(logprobabilities.argmax(-1))
                y_score_list.append(logprobabilities.exp())

        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list), torch.cat(y_score_list)

# %%

trainer.test(model1, dataloader_train_all)
# %%
losses
 # %%
losses, y_true, y_pred, y_score = test_epoch( model1, torch.nn.CrossEntropyLoss(), dataloader_test_all, device )
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print(classification_report(y_true.cpu(), y_pred.cpu()))
print("OA:",round(accuracy_score(y_true, y_pred),2))
# %%

logprobabilities
# %%

losses2, y_true2, y_pred2, y_score2 = test_epoch( model2, torch.nn.CrossEntropyLoss(), dataloader18_test, device )
losses3, y_true3, y_pred3, y_score3 = test_epoch( model3, torch.nn.CrossEntropyLoss(), dataloader18_test, device )

print("Second Experiment:")
print(classification_report(y_true2.cpu(), y_pred2.cpu()))
print("OA:",round((y_true2, y_pred2)))

print("Third Experiment:")
print(classification_report(y_true3.cpu(), y_pred3.cpu()))
print("OA:",round(accuracy_score(y_true3, y_pred3)))

#https://towardsdatascience.com/pytorch-lightning-making-your-training-phase-cleaner-and-easier-845c4629445b

#find learning rate
"""lr_finder = trainer.tuner.lr_find(model1, 
                        min_lr=0.0005, 
                        max_lr=0.005,
                        mode='linear')
# Plots the optimal learning rate
fig = lr_finder.plot(suggest=True)
fig.show()
# %%
# find batch size
trainer.tune(model1)"""
# %%
"""
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

confusion = pd.DataFrame()
confusion['y_pred'] = y_pred
confusion['y_test'] = y_test.values
#printConfusionResults(confusion)
# %%



# %%



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

#sample some examples from 2018
_2018 = train_RF[(train_RF.Year == 2018)].copy()
samples = pd.DataFrame()
for i in range(1,7):
    sample = _2018[(_2018.NC == i)].sample(1)
    samples = pd.concat([samples,sample],axis=0)

subset = train_RF[train_RF.Year == 2018].sample(30).copy()
#######

train_1617 = train_RF[train_RF.Year != 2018].copy()
train  = pd.concat([train_1617,samples],axis = 0)
X_train = train[train.columns[train.columns.str.contains(band2, na=False)]]
y_train = train['NC']

test_RF = train_RF[train_RF.Year == 2018].copy()
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


# %%

#X = train_RF[train_RF.columns[train_RF.columns.str.contains(band2, na=False)]]
#y = train_RF['NC']

# %%
samples

# %%
train_1617.describe()

