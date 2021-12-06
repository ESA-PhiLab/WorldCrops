# %%
import os
import math
import matplotlib.pyplot as plt
import matplotlib.offsetbox as osb
from matplotlib import rcParams as rcp
import numpy as np
import tqdm
import pandas as pd

import torchvision.transforms.functional as functional
from sklearn import random_projection
import pytorch_lightning as pl

from torchvision.datasets import CIFAR10
from torchvision import transforms

import sys
sys.path.append('./model')
sys.path.append('..')

import model
from model import *
from processing import *
from tsai.all import *
# %%

pwd
# %%

# %%


# %%
# Prepare datasets
cifar = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())

input = np.arange(1,8).reshape(-1,1)
input = torch.tensor(input, dtype=torch.float)
own = MyDataSet(input, 3)


#load data for bavaria
bavaria_train = pd.read_excel(
    "../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx")
bavaria_test = pd.read_excel(
    "../../data/cropdata/Bavaria/sentinel-2/Test_bavaria.xlsx")

bavaria_reordered = pd.read_excel(
    '../../data/cropdata/Bavaria/sentinel-2/data2016-2018.xlsx', index_col=0)
bavaria_test_reordered = pd.read_excel(
    '../../data/cropdata/Bavaria/sentinel-2/TestData.xlsx', index_col=0)


train = utils.clean_bavarian_labels(bavaria_train)
test = utils.clean_bavarian_labels(bavaria_test)
# %%
test.NC.unique()
# %%
#tsai needs (samples,features,timestemps)
# only for tsai !


n_fields = len(train.id.unique())
n_days = 14
feature_list = train.columns[train.columns.str.contains("B", na=False)].tolist()

train = train.sort_values(by=['id', 'Date'])
arr = train[feature_list].values.reshape(n_fields, n_days, len(feature_list) )

#switch timestemps and features axis -> (samples,features,timestemps)
X = np.swapaxes(arr, 1, 2)
y = train['NC'].values.reshape(n_fields, n_days, 1 )
y = y[:,0,0]
 
splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=True)
tfms = [None, TSClassification()]
batch_tfms = [TSStandardize(by_sample=True)]
check_data(X, y, splits)
dls100 = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)

# %%




# %%
# parameters
input_dim = 32 * 32 * 3
num_classes = 10

# implement MLP with lightning module
#https://www.machinecurve.com/index.php/2021/01/26/creating-a-multilayer-perceptron-with-pytorch-and-lightning/

pl.seed_everything(42)
#model = MLP(input_dim, num_classes)
#model = RNN_LM(3, 1, 128)


# %%
#clsmembers = inspect.getmembers(sys.modules['model'], inspect.isclass)
#clsmembers



# %%
# train breizhcrops
import torch
import tqdm
import breizhcrops as bc
model = bc.models.TransformerModel(num_classes=7)
model.train()
dataset = bc.BreizhCrops("belle-ile")
feature_list = train.columns[train.columns.str.contains('B')]
ts_data = TimeSeriesDataSet(train, feature_list.tolist(), 'NC')
dataloader = DataLoader(ts_data,batch_size=3,shuffle=True,drop_last=False,num_workers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(5):
  with tqdm.tqdm(enumerate(dataloader),total=(len(dataloader))) as pbar:
    for idx, batch in pbar:
      optimizer.zero_grad()    
      X,y = batch
      #print(X.size())
      y_pred = model(X)
      loss = criterion(y_pred, y)
      loss.backward()
      optimizer.step()
      pbar.set_description(f"idx {idx}: loss {loss:.2f}")

# %%
test.tail()
# %%
#test model
feature_list = test.columns[test.columns.str.contains('B')]
ts_testdata = TimeSeriesDataSet(test, feature_list.tolist(), 'NC')
dataloader_test = DataLoader(ts_testdata,batch_size=3,shuffle=True,drop_last=False,num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_epoch(model, criterion, dataloader, device):
    model.eval()
    with torch.no_grad():
        losses = list()
        y_true_list = list()
        y_pred_list = list()
        y_score_list = list()

        with tqdm.tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, batch in iterator:
                x, y_true = batch
                logprobabilities = model.forward(x.to(device))
                loss = criterion(logprobabilities, y_true.to(device))
                iterator.set_description(f"test loss={loss:.2f}")
                losses.append(loss)
                y_true_list.append(y_true)
                y_pred_list.append(logprobabilities.argmax(-1))
                y_score_list.append(logprobabilities.exp())

        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list), torch.cat(y_score_list)

import sklearn 
losses, y_true, y_pred, y_score = test_epoch( model, torch.nn.CrossEntropyLoss(), dataloader_test, device )
print(sklearn.metrics.classification_report(y_true.cpu(), y_pred.cpu()))
# %%
# train own implementation
import torch
import tqdm
model = Attention(num_classes = 7)
model.train()

feature_list = train.columns[train.columns.str.contains('B')]
ts_data = TimeSeriesDataSet(train, feature_list.tolist(), 'NC')
dataloader = DataLoader(ts_data,batch_size=3,shuffle=True,drop_last=False,num_workers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(5):
  with tqdm.tqdm(enumerate(dataloader),total=(len(dataloader))) as pbar:
    for idx, batch in pbar:
      optimizer.zero_grad()    
      X,y = batch
      #print(X.size())
      y_pred = model(X)
      loss = criterion(y_pred, y)
      loss.backward()
      optimizer.step()
      pbar.set_description(f"idx {idx}: loss {loss:.2f}")
# %%
losses, y_true, y_pred, y_score = test_epoch( model, torch.nn.CrossEntropyLoss(), dataloader_test, device )
print(sklearn.metrics.classification_report(y_true.cpu(), y_pred.cpu()))
# %%
model
# %%
#train own again
import breizhcrops as bc
dataset = bc.BreizhCrops("belle-ile")
feature_list = train.columns[train.columns.str.contains('B')]
ts_data = TimeSeriesDataSet(train, feature_list.tolist(), 'NC')
dataloader = DataLoader(ts_data,batch_size=3,shuffle=True,drop_last=False,num_workers=2)

model1 = Attention_LM(num_classes = 7)
model1.train()
feature_list = train.columns[train.columns.str.contains('B')]
ts_data = TimeSeriesDataSet(train, feature_list.tolist(), 'NC')
#dataloader = DataLoader(ts_data,batch_size=3,shuffle=True,drop_last=False,num_workers=2)

trainer = pl.Trainer(auto_scale_batch_size='power', gpus=0, deterministic=True, max_epochs=5)
trainer.fit(model1, dataloader)

# %%
import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def test_epoch(model, criterion, dataloader, device):
    model.eval()
    with torch.no_grad():
        losses = list()
        y_true_list = list()
        y_pred_list = list()
        y_score_list = list()

        with tqdm.tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, batch in iterator:
                x, y_true = batch
                logprobabilities = model.forward(x.to(device))
                loss = criterion(logprobabilities, y_true.to(device))
                iterator.set_description(f"test loss={loss:.2f}")
                losses.append(loss)
                y_true_list.append(y_true)
                y_pred_list.append(logprobabilities.argmax(-1))
                y_score_list.append(logprobabilities.exp())

        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list), torch.cat(y_score_list)

losses, y_true, y_pred, y_score = test_epoch( model1, torch.nn.CrossEntropyLoss(), dataloader_test, device )
print(sklearn.metrics.classification_report(y_true.cpu(), y_pred.cpu()))
# %%

report = pd.DataFrame(sklearn.metrics.classification_report(y_true = y_true.cpu(), y_pred = y_pred.cpu(), output_dict=True)).transpose()
report.to_csv('results/classification.csv', index= True)
# %%
report
# %%
