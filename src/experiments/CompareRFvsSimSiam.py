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
# %%

# %%
#rewrite the 'id' as we deleted one class
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
len(test.id.unique())
test.NC.unique()
# %%
years = [2016,2017,2018]
train = utils.augment_df(train, years)

 # %%
# %%
############################################################################
# SimSiam
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
def plot_firstinbatch(aug_x1, aug_x2, timeseries, labels):
    fig, axs = plt.subplots(3)
    fig.suptitle('Augmentation')
    axs[0].plot(timeseries[0].numpy().reshape(14, 13), "-")
    axs[1].plot(aug_x1[0].numpy().reshape(14, 13), "-")
    axs[2].plot(aug_x2[0].numpy().reshape(14, 13), "-")
    print("Crop type:", labels[0])
    plt.show()

dataiter = iter(dataloader_train)
(x1,x2),x, y = next(dataiter)
plot_firstinbatch(x1, x2, x, y)
# %%
x1.shape

# %%
gpus = 1 if torch.cuda.is_available() else 0
seed = 1
# seed torch and numpy
torch.manual_seed(0)
np.random.seed(0)


epochs = 2
#input size (timesteps x channels)
input_size = 14
num_ftrs = 64
proj_hidden_dim =14
pred_hidden_dim =14
out_dim =14
# scale the learning rate
lr = 0.05 * batch_size / 256

#model definition
transformer = Attention(num_classes = 6)
backbone  = nn.Sequential(*list(transformer.children())[:-1])
model = SimSiam(backbone,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim)

#model.prediction = lightly.models.modules.heads.SimSiamPredictionHead(14, 14, 14)

# %%
#outputs do not collapse wenn sich nache 1/math.sqrt(outdim) liegen
#collapse can be observed by the mini-mum possible loss and the constant outputs
1/math.sqrt(14)

# %%
#train like in description
criterion = lightly.loss.NegativeCosineSimilarity()
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
    for (x0,x1) ,_, _ in dataloader_train:
      
        x0 = x0.to(device)
        x1 = x1.to(device)
        (z0, p0), (z1, p1) = model(x0,x1)

        #print(z1.shape,p1.shape)
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
class SimSiam_LM(pl.LightningModule):
    def __init__(self, backbone = nn.Module, num_ftrs=64, proj_hidden_dim=14, 
    pred_hidden_dim=14, out_dim=14, lr=0.06, weight_decay=5e-4,momentum=0.9,epochs = 10):
        super().__init__()
        
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.backbone = backbone
        self.model_type = 'SimSiam_LM'
        self.projection = lightly.models.modules.heads.ProjectionHead([
            (num_ftrs, proj_hidden_dim, nn.BatchNorm1d(proj_hidden_dim), nn.ReLU()),
            (proj_hidden_dim, out_dim, nn.BatchNorm1d(out_dim), None)
        ])
        self.prediction = SimSiamPredictionHead(out_dim,pred_hidden_dim,out_dim)

        self.ce = lightly.loss.NegativeCosineSimilarity()

        self.avg_loss = 0.
        self.avg_output_std = 0.
        

    def forward(self, x0, x1):
        f0 = self.backbone(x0)
        f1 = self.backbone(x1)

        z0 = self.projection(f0)
        z1 = self.projection(f1)

        p0 = self.prediction(z0)
        p1 = self.prediction(z1)

        z0 = z0.detach()
        z1 = z1.detach()
        return (z0, p0),(z1, p1)


    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        (z0, p0),(z1, p1) = self.forward(x0,x1)

        loss = 0.5 * (self.ce(z0, p1) + self.ce(z1, p0))
        self.log('train_loss_ssl', loss)
        #output = p0.detach()
        return {'loss':loss}

    '''def training_step_end(self, batch_parts):
        #print(batch_parts['p0'], type(batch_parts['p0']))
        output = batch_parts['p0']
        loss = batch_parts['loss']
        output = torch.nn.functional.normalize(output, dim=1)
        output_std = torch.std(output, 0)
        output_std = output_std.mean()
        
        # use moving averages to track the loss and standard deviation
        w = 0.9
        self.avg_loss = w * self.avg_loss + (1 - w) * loss
        self.avg_output_std = w * self.avg_output_std + (1 - w) * output_std.item()
        return {'loss':self.avg_loss,'avg_output_std': self.avg_output_std}
    
    def training_epoch_end(self, training_step_outputs):

        collapse_level = max(0., 1 - math.sqrt(14) * self.avg_output_std)
        self.log('loss', round(self.avg_loss,2), prog_bar=True)
        self.log('Collapse Level', round(collapse_evel,2), prog_bar=True)'''


    def validation_step(self, val_batch, batch_idx):
        (x0, x1), _, _ = batch
 
        (z0, p0),(z1, p1) = self.forward(x0,x1)
        loss = 0.5 * (self.ce(z0, p1) + self.ce(z1, p0))
        self.log('val_loss', loss)
        return {"loss":loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,
                                momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        return [optimizer], [scheduler]

model2 = SimSiam_LM(backbone,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim)
trainer = pl.Trainer(auto_scale_batch_size='power', gpus=0, deterministic=True, max_epochs=5)
trainer.fit(model2, dataloader_train)


# %% 
_weights = model2.backbone
torch.save(_weights.state_dict(), '../model/pretrained/attention_weights.pth')

# %% 


# %%  
#%load_ext tensorboard
#%tensorboard --logdir {lightning_logs}

# %%  
#take trained backbone / add new linear prediction head /train with 10%
model2
# %%  
model = model2.backbone
model
#model2.prediction = SimSiamPredictionHead(out_dim,pred_hidden_dim,out_dim)

modules = []
modules.append(nn.Linear(64, 6))

model(*modules)
model
# %%  
model2
# %%    
embeddings = []
targets = []
# disable gradients for faster calculations
model.eval()
with torch.no_grad():
    for x, y in dataloader_test:
        #print(x.shape)
        targets.append(y)
        x = x.to(device)
        # embed the images with the pre-trained backbone
        y = model.backbone(x).flatten(start_dim=1)
        print(y.shape)
        embeddings.append(y)
        

# concatenate the embeddings and convert to numpy
embeddings = torch.cat(embeddings, dim=0)
embeddings = embeddings.cpu().numpy()

# %%

# %%
#hier gehts weiter. Ich will einen plot mit den timeseries haben
#und deren ähnlichste nachbarn mit angabe welche crop types dargestellt werden
#
# ransform the 13d timeseries to a two-dimensional
# vector space using a random Gaussian projection
from sklearn import random_projection
projection = random_projection.GaussianRandomProjection(n_components=2)
embeddings_2d = projection.fit_transform(embeddings)

# %%
# normalize the embeddings to fit in the [0, 1] square
M = np.max(embeddings_2d, axis=0)
m = np.min(embeddings_2d, axis=0)
embeddings_2d = (embeddings_2d - m) / (M - m)

# %%
def get_scatter_plot_with_thumbnails():
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
    

    # plot image overlays
    for idx in shown_images_idx:
        #print(embeddings_2d[idx])
        #thumbnail_size = int(rcp['figure.figsize'][0] * 2.)
        #path = os.path.join(path_to_data, filenames[idx])
        #img = Image.open(path)
        #img = functional.resize(img, thumbnail_size)
        #img = np.array(img)
        #img_box = osb.AnnotationBbox(
        #    osb.OffsetImage(img, cmap=plt.cm.gray_r),
        #    embeddings_2d[idx],
        #    pad=0.2,
        #)
        circle = plt.Circle((embeddings_2d[idx][0], embeddings_2d[idx][1]), 0.02, color='r')
        ax.add_artist(circle)

    # set aspect ratio
    ratio = 1. / ax.get_data_ratio()
    ax.set_aspect(ratio, adjustable='box')

get_scatter_plot_with_thumbnails()
# %%
#pendigits = sklearn.datasets.load_digits()
#mnist = sklearn.datasets.fetch_openml('mnist_784')
#fmnist = sklearn.datasets.fetch_openml('Fashion-MNIST')
pendigits.target
# %%
mapper = umap.UMAP().fit(embeddings_2d)
umap.plot.points(mapper, labels=pendigits.target, color_key_cmap='Paired', background='black')

# %%
mnist.data.shape
# %%
def plot_one_example(timeseries, labels):
    fig, axs = plt.subplots(5,figsize=(10,15))
    fig.suptitle('Augmentation & Nearest Neighbors')
    axs[0].plot(x[0].numpy().reshape(14, 13), "-")
    print("Crop type:", labels[0])
    plt.show()

dataiter = iter(dataloader_test)
x, y = next(dataiter)
plot_one_example(x, y)


# %%
