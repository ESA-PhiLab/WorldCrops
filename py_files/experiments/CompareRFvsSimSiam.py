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

#tsai could be helpful
from tsai.all import *
computer_setup()
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

train = utils.clean_bavarian_labels(bavaria_train)
test = utils.clean_bavarian_labels(bavaria_test)
# %%
test
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
#printConfusionResults(confusion)
# %%
confusion

# %%
############################################################################
# SimSiam
############################################################################
# Custom DataSet with augmentation
# augmentation needs to be extended

feature_list = train.columns[train.columns.str.contains('B')]
ts_dataset = TimeSeriesAugmented(train, feature_list.tolist(), 'NC')
ts_dataset_test = TimeSeriesAugmented(test, feature_list.tolist(), 'NC')

dataloader_train = torch.utils.data.DataLoader(
    ts_dataset, batch_size=3, shuffle=True,drop_last=False, num_workers=2)
dataloader_test = torch.utils.data.DataLoader(
    ts_dataset_test, batch_size=3, shuffle=True,drop_last=False, num_workers=2)

# %%

# %%
def plot_firstinbatch(aug_x1, aug_x2, timeseries, labels):
    fig, axs = plt.subplots(3)
    fig.suptitle('Augmentation')

    #x  = timeseries[:,bands_idxs]
    axs[0].plot(timeseries[0].numpy().reshape(14, 13), "-")
    axs[1].plot(aug_x1[0].numpy().reshape(14, 13), "-")
    axs[2].plot(aug_x2[0].numpy().reshape(14, 13), "-")

    print("Crop type:", labels)
    plt.show()

for i, data in enumerate(dataloader_train):
    if (i > 0):
        break
    (aug_x1, aug_x2), timeseries, labels = data
    print (aug_x1.shape,labels.shape)
    plot_firstinbatch(aug_x1, aug_x2, timeseries, labels)

# %%


# %%
cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
dataset = lightly.data.LightlyDataset.from_torch_dataset(cifar10)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = lightly.data.SimCLRCollateFunction(input_size=32)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

# %%
gpus = 1 if torch.cuda.is_available() else 0
seed = 1
# seed torch and numpy
torch.manual_seed(0)
np.random.seed(0)

batch_size=3
epochs = 5
#input size (timesteps x channels)
input_size = 14
num_ftrs = 28
proj_hidden_dim =7
pred_hidden_dim = 10
out_dim =7

transformer = Attention_LM(num_classes = 7)
# without pretraining
backbone  = nn.Sequential(*list(transformer.children())[:-1])

# scale the learning rate
lr = 0.05 * batch_size / 256
model = SimSiam(backbone, num_ftrs = num_ftrs, proj_hidden_dim =proj_hidden_dim, pred_hidden_dim = pred_hidden_dim, out_dim = out_dim,lr=lr)

# %%
backbone
# %%
model.projection_mlp = lightly.models.modules.heads.ProjectionHead([
    (num_ftrs, proj_hidden_dim, nn.BatchNorm1d(proj_hidden_dim), nn.ReLU()),
    (proj_hidden_dim, out_dim, nn.BatchNorm1d(out_dim), None)
])
# %%
SimSiam
# %%
trainer = pl.Trainer(max_epochs=5, gpus=0, deterministic=True)
trainer.fit(model=model, train_dataloaders=dataloader_train)

# %%
model1 = Attention_LM(num_classes = 7)
model1.train()
feature_list = train.columns[train.columns.str.contains('B')]
ts_data = TimeSeriesDataSet(train, feature_list.tolist(), 'NC')
dataloader = DataLoader(ts_data,batch_size=3,shuffle=True,drop_last=False,num_workers=2)

trainer = pl.Trainer(auto_scale_batch_size='power', gpus=0, deterministic=True, max_epochs=5)
trainer.fit(model1, dataloader)

# %%
model1
# %%
#train like in description
model1 = Attention_LM(num_classes = 7)
model1.train()
feature_list = train.columns[train.columns.str.contains('B')]
ts_data = TimeSeriesAugmented(train, feature_list.tolist(), 'NC')
dataloader2 = DataLoader(ts_data,batch_size=3,shuffle=True,drop_last=False,num_workers=2)
criterion = lightly.loss.NTXentLoss()
lr = 0.05 * batch_size / 256
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=5e-4
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

avg_loss = 0.
avg_output_std = 0.
for e in range(epochs):

    for (x0,x1) ,_, _ in dataloader2:
      
        # move images to the gpu
        #aug[0] = torch.unsqueeze(aug[0], 1)
        #aug[0] = aug[0].permute(0,3,2,1)
        #aug[1] = torch.unsqueeze(aug[1], 1)
        #aug[1] = aug[1].permute(0,3,2,1)

        #x0 = aug[0].to(device)
        #x1 = aug[1].to(device)
        print(x1.shape)

        #x0 = torch.unsqueeze(X, 1)
        #x1 = torch.unsqueeze(X, 1)
        x0 = x0.to(device)
        x1 = x1.to(device)

        print(x0.shape)
        # run the model on both transforms of the images
        # the output of the simsiam model is a y containing the predictions
        # and projections for each input x
        z0, p0 = model(x0)
        
        z1, p1 = model(x1)

        loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # calculate the per-dimension standard deviation of the outputs
        # we can use this later to check whether the embeddings are collapsing
        output = p0.detach()
        output = torch.nn.functional.normalize(output, dim=1)

        output_std = torch.std(output, 0)
        output_std = output_std.mean()

        # use moving averages to track the loss and standard deviation
        w = 0.9
        avg_loss = w * avg_loss + (1 - w) * loss.item()
        avg_output_std = w * avg_output_std + (1 - w) * output_std.item()

    # the level of collapse is large if the standard deviation of the l2
    # normalized output is much smaller than 1 / sqrt(dim)
    collapse_level = max(0., 1 - math.sqrt(out_dim) * avg_output_std)
    # print intermediate results
    print(f'[Epoch {e:3d}] '
        f'Loss = {avg_loss:.2f} | '
        f'Collapse Level: {collapse_level:.2f} / 1.00')
# %%
dataiter = iter(dataloader2)
(x1,x2),x, y = next(dataiter)
print(x1.shape)
# %%
