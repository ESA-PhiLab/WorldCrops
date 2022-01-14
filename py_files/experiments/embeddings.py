# %%
import sklearn.datasets
import pandas as pd
import numpy as np
import umap
import umap.plot
import lightly

import torch
import seaborn as sns
import numpy as np

import sys
sys.path.append('./model')
sys.path.append('..')

from model import *
from processing import *
import math

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# %%
# set the path to the dataset
path_to_data = '../../data/simpsons_100/'
batch_size = 16
input_size = 256
num_workers = 0
seed = 1
epochs = 50

# seed torch and numpy
torch.manual_seed(0)
np.random.seed(0)

# define the augmentations for self-supervised learning
collate_fn = lightly.data.ImageCollateFunction(
    input_size=input_size,
    # require invariance to flips and rotations
    hf_prob=0.5,
    vf_prob=0.5,
    rr_prob=0.5,
    # satellite images are all taken from the same height
    # so we use only slight random cropping
    min_scale=0.5,
    # use a weak color jitter for invariance w.r.t small color changes
    cj_prob=0.2,
    cj_bright=0.1,
    cj_contrast=0.1,
    cj_hue=0.1,
    cj_sat=0.1,
)
# create a torchvision transformation for embedding the dataset after training
# here, we resize the images to match the input size during training and apply
# a normalization of the color channel based on statistics from imagenet
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])


dataset_train_simsiam = lightly.data.LightlyDataset(
    input_dir=path_to_data
)

# create a dataloader for training
dataloader_train_simsiam = torch.utils.data.DataLoader(
    dataset_train_simsiam,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)

# create a lightly dataset for embedding
dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_data,
    transform=test_transforms
)

# create a dataloader for embedding
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)


bavaria_reordered = pd.read_excel(
    '../../data/cropdata/Bavaria/sentinel-2/data2016-2018.xlsx', index_col=0)
bavaria_test_reordered = pd.read_excel(
    '../../data/cropdata/Bavaria/sentinel-2/TestData.xlsx', index_col=0)

train = utils.clean_bavarian_labels(bavaria_reordered)
test = utils.clean_bavarian_labels(bavaria_test_reordered)

#################################################
#main
#################################################
#https://umap-learn.readthedocs.io/en/latest/transform.html?highlight=classification%20accuracy
#https://umap-learn.readthedocs.io/en/latest/aligned_umap_basic_usage.html

#mapper = umap.UMAP().fit(embeddings_2d)
#umap.plot.points(mapper, color_key_cmap='Paired', background='black')
# %%
test = train.columns.values.tolist()
test.remove('NC')
test.remove('area')
#test.remove('Year')

# %%
train = train[train.NC != 0]
train.NC.unique()
# %%
train.describe()
# %%
#sns.pairplot(train, hue='NC')
# %%
reducer = umap.UMAP()
_data = train[test].values
scaled_data = StandardScaler().fit_transform(_data)

embedding = reducer.fit_transform(scaled_data)
embedding.shape
# %%
train.NC.values
# %%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
colors = ['b', 'c', 'y', 'm', 'r','r']
values = train.NC.values
classes = ['A', 'B', 'C', 'D', 'E', 'F']
colours = ListedColormap(colors)

ax = plt.subplot(111)
ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=values,  s = 0.4)

#plt.legend()
plt.legend(handles=scatter.legend_elements()[0])
plt.title('UMAP Representation', fontsize=24)
# %%

# %%


# %%
plt.savefig('foo.png')
# %%
from io import BytesIO
from PIL import Image
import base64
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10

output_notebook()

# %%

train.NC
# %%
digits_df = pd.DataFrame(embedding, columns=('x', 'y'))
digits_df['digit'] = [str(x) for x in train.NC]
digits_df['year'] = [str(x) for x in train.Year]

datasource = ColumnDataSource(digits_df)
color_mapping = CategoricalColorMapper(palette=["red", "blue",'orange','black','grey','yellow'], factors=["1", "2", "3", "4", "5", "6"])

plot_figure = figure(
    title='UMAP projection dataset',
    plot_width=600,
    plot_height=600,
    tools=('pan, wheel_zoom, reset')
)

plot_figure.add_tools(HoverTool(tooltips="""
<div>
    <div>
        <span style='font-size: 16px; color: #224499'>Digit:</span>
        <span style='font-size: 18px'>@digit</span>
        <span style='font-size: 18px'>@year</span>
    </div>
</div>
"""))

plot_figure.circle(
    'x',
    'y',
    source=datasource,
    color=dict(field='digit', transform=color_mapping),
    line_alpha=0.6,
    fill_alpha=0.6,
    size=4
)
show(plot_figure)
# %%
test = digits_df.copy()
test['digit'] = test['digit'].astype(int)
mymap = {1:'Potato', 2:'Barley', 3:'Corn', 4:'Rapeseed', 5:'Wheat', 6:'Sugar beat'}
test = test.applymap(lambda x: mymap.get(x) if x in mymap else x)

import seaborn as sns
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,7))
sns.scatterplot(data=test, hue='digit', x='x', y='y',ax=ax1)
ax1.legend(loc='upper right', fontsize=16)
sns.scatterplot(data=test, hue='year', x='x', y='y',ax=ax2)
ax2.legend(loc='upper right', fontsize=16)
#plt.savefig('scattplt.legend(loc=2)
fig.savefig('embeddings.png')

ax1.set_ylabel('')
ax1.set_xlabel('')
ax2.set_ylabel('')
ax2.set_xlabel('')
plt.show()

# %%
from io import BytesIO
from PIL import Image
import base64

def embeddable_image(data):
    img_data = 255 - 15 * data.astype(np.uint8)
    image = Image.fromarray(img_data, mode='L').resize((64, 64), Image.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()
# %%
plt.scatter(u[:,0], u[:,1], c=data)
plt.title('UMAP embedding of random colours');
# %%
'''Lets do some classification stuf here'''
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(train,
                                                    train.NC,
                                                    shuffle =True,
                                                    stratify=train.NC,
                                                    random_state=42)

# %%
from sklearn.svm import SVC
svc = SVC().fit(X_train[test], y_train)
svc.score(X_test[test], y_test)
# %%
trans = umap.UMAP(n_neighbors=6, random_state=42).fit(X_train[test])
# %%
plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 5, c=y_train, cmap='Spectral')
plt.title('Embedding of the training set by UMAP', fontsize=24);
# %%
svc = SVC().fit(trans.embedding_, y_train)
# %%
#time test_embedding = trans.transform(X_test[test])
plt.scatter(test_embedding[:, 0], test_embedding[:, 1], s= 5, c=y_test, cmap='Spectral')
plt.title('Embedding of the test set by UMAP', fontsize=24);
# %%
svc.score(trans.transform(X_test[test]), y_test)
# %%
X_test[test].head()
# %%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier().fit(trans.embedding_, y_train)
knn.score(trans.transform(X_test[test]), y_test)
# %%
