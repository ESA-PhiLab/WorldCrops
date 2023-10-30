# %%
# compare the crop type classification of RF and SimSiam

import glob
import re

import lightly
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
from PIL import Image
from torchsummary import summary

import selfsupervised as ssl
from selfsupervised.processing import utils

utils.seed_torch()

with open("../../config/iceberg/param_config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

batch_size_pre = cfg["pretraining"]['batch_size']
batch_size_fine = cfg["finetuning"]['batch_size']
input_size = cfg["pretraining"]['input_size']

num_ftrs = cfg["pretraining"]['num_ftrs']
proj_hidden_dim = cfg["pretraining"]['proj_hidden_dim']
pred_hidden_dim = cfg["pretraining"]['pred_hidden_dim']
out_dim = cfg["pretraining"]['out_dim']

num_workers = 0
path_to_data = '../../data/Anne/'
# %%
# %%

# zp = ZipFile('../../data/Anne/OneDrive_1_17-08-2022.zip')
# zp.extractall(path = '../../data/Anne')
# zp.close()

x_train = []
y_train = []
train_mask = []

for filename in sorted(glob.glob(path_to_data + 'PNG_train/S1*.png')):

    # split filename to find the matching files
    name = re.search('/S1_(.*).png', filename).group(1)
    name_part1 = re.search('/S1_(.*)99_', filename).group(1)
    name_part2 = re.search('_99_(.*).png', filename).group(1)

    # load input image
    im = Image.open(filename)
    x_train.append(np.array(im))

    # load corresponding ground truth
    filename2 = path_to_data + 'GT_train/GT_' + name + '.png'
    gt = Image.open(filename2)
    y_train.append(np.array(gt))

    # load corresponding mask (no satellite coverage)
    filename3 = path_to_data + 'Mask_train/NaN_mask_' + \
        name_part1 + name_part2 + '.png'
    mask = Image.open(filename3)
    train_mask.append(np.array(mask))

x_train = np.array(x_train)
y_train = np.array(y_train)
train_mask = np.array(train_mask)

print('Training data sizes (should be 161 x 256 x 256 for all of them)')
print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(train_mask))

x_val = []
y_val = []
val_mask = []

for filename in sorted(glob.glob(path_to_data + 'PNG_val/S1*.png')):

    # split filename to find the matching files
    name = re.search('/S1_(.*).png', filename).group(1)
    name_part1 = re.search('/S1_(.*)99_', filename).group(1)
    name_part2 = re.search('_99_(.*).png', filename).group(1)

    # load input image
    im = Image.open(filename)
    x_val.append(np.array(im))

    # load corresponding ground truth
    filename2 = path_to_data + 'GT_val/GT_' + name + '.png'
    gt = Image.open(filename2)
    y_val.append(np.array(gt))

    # load corresponding mask (no satellite coverage)
    filename3 = path_to_data + 'Mask_val/NaN_mask_' + name_part1 + name_part2 + '.png'
    mask = Image.open(filename3)
    val_mask.append(np.array(mask))

x_val = np.array(x_val)
y_val = np.array(y_val)
val_mask = np.array(val_mask)

print('Validation data sizes (should be 26 x 256 x 256 for all of them)')
print(np.shape(x_val))
print(np.shape(y_val))
print(np.shape(val_mask))

x_test = []
y_test = []
test_mask = []

for filename in sorted(glob.glob(path_to_data + 'PNG_test/S1*.png')):

    # split filename to find the matching files
    name = re.search('/S1_(.*).png', filename).group(1)
    name_part1 = re.search('/S1_(.*)99_', filename).group(1)
    name_part2 = re.search('_99_(.*).png', filename).group(1)

    # load input image
    im = Image.open(filename)
    x_test.append(np.array(im))

    # load corresponding ground truth
    filename2 = path_to_data + 'GT_test/GT_' + name + '.png'
    gt = Image.open(filename2)
    y_test.append(np.array(gt))

    # load corresponding mask (no satellite coverage)
    filename3 = path_to_data + 'Mask_test/NaN_mask_' + name_part1 + name_part2 + '.png'
    mask = Image.open(filename3)
    test_mask.append(np.array(mask))

x_test = np.array(x_test)
y_test = np.array(y_test)
test_mask = np.array(test_mask)

print('Test data sizes (should be 24 x 256 x 256 for all of them)')
print(np.shape(x_test))
print(np.shape(y_test))
print(np.shape(test_mask))

x_unlabeled = []
unlabeled_mask = []

for filename in sorted(glob.glob(path_to_data + 'PNG_unlabeled/S1*.png')):

    # split filename to find the matching files
    name_part1 = re.search('/S1_(.*)99_', filename).group(1)
    name_part2 = re.search('_99_(.*).png', filename).group(1)

    # load input image
    im = Image.open(filename)
    x_unlabeled.append(np.array(im))

    # load corresponding mask (no satellite coverage)
    filename3 = path_to_data + 'Mask_unlabeled/NaN_mask_' + \
        name_part1 + name_part2 + '.png'
    mask = Image.open(filename3)
    unlabeled_mask.append(np.array(mask))

x_unlabeled = np.array(x_unlabeled)
unlabeled_mask = np.array(unlabeled_mask)

x_train_unsup = np.concatenate((x_train, x_unlabeled), axis=0)
train_unsup_mask = np.concatenate((train_mask, unlabeled_mask), axis=0)

print(
    'Unsupervised training data sizes (Labeled training data + unlabeled data; should be 275 x 256 x 256 for all of them)'
)
print(np.shape(x_train_unsup))
print(np.shape(train_unsup_mask))

# also ich brauche für self-supervsied nicht unbedingt die labels
# sprich ich kann lightly dafür verwenden
#######################
# für finetuning brauch ich die labels

# x_train_unsup.shape

# path_to_data + 'PNG_unlabeled/S1*.png'
# path_to_data + 'PNG_train/S1*.png'

################################################################
# Augmentations + custom dataset from lightly
# https://docs.lightly.ai/tutorials/package/tutorial_simsiam_esa.html
###################################################################

# define the augmentations for self-supervised learning
collate_fn = lightly.data.ImageCollateFunction(
    input_size=256,
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

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
])

dataset_train = lightly.data.LightlyDataset(input_dir=path_to_data +
                                            'PNG_train/',
                                            transform=test_transforms)
dataset_unlabeled = lightly.data.LightlyDataset(input_dir=path_to_data +
                                                'PNG_unlabeled/',
                                                transform=test_transforms)

# create a lightly dataset for embedding
dataset_test = lightly.data.LightlyDataset(
    input_dir='../../data/Anne/PNG_test', transform=test_transforms)

train_dev = torch.utils.data.ConcatDataset([dataset_train, dataset_unlabeled])

dataloader_train_unsupervised = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=10,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers)

# create a dataloader for embedding
dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=batch_size_pre,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=num_workers)

tensor_x = torch.Tensor(x_train[:,
                                np.newaxis, :, :])  # transform to torch tensor
tensor_y = torch.Tensor(y_train[:, np.newaxis, :, :])

my_dataset = torch.utils.data.TensorDataset(tensor_x,
                                            tensor_y)  # create your datset
my_dataloader = torch.utils.data.DataLoader(
    my_dataset,
    batch_size=10,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers)  # create your dataloader


# %%
class depthwise_separable_conv(nn.Module):

    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin,
                                   nin * kernels_per_layer,
                                   kernel_size=3,
                                   padding='same',
                                   groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer,
                                   nout,
                                   kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvDown(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvDown, self).__init__()

        self.conv_block = nn.Sequential(
            depthwise_separable_conv(input_dim, 1, output_dim), nn.ReLU(),
            nn.Dropout(p=0.3),
            depthwise_separable_conv(output_dim, 1, output_dim), nn.ReLU())

    def forward(self, x):
        return self.conv_block(x)


class ResidualDown(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ResidualDown, self).__init__()

        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2), )

    def forward(self, x):
        xpad = F.pad(x, [0, 1, 0, 1], mode='replicate')
        return self.conv_skip(xpad)


class ResidualBridge(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ResidualBridge, self).__init__()

        self.conv_block = nn.Sequential(
            depthwise_separable_conv(input_dim, 1, output_dim), nn.ReLU(),
            nn.Dropout(p=0.3),
            depthwise_separable_conv(output_dim, 1, output_dim), nn.ReLU())

    def forward(self, x):
        return self.conv_block(x)


class ResidualUp(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ResidualUp, self).__init__()

        self.conv_skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(input_dim, output_dim, kernel_size=1, padding='same'))

    def forward(self, x):
        return self.conv_skip(x)


class ConvUp(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvUp, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(input_dim + input_dim // 2,
                      output_dim,
                      kernel_size=3,
                      padding='same'), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding='same'),
            nn.ReLU(), nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, x):
        return self.conv_block(x)


class ResUnetEncoder3(nn.Module):

    def __init__(self, channel=1, dropout=0.3, filters=None):
        super(ResUnetEncoder3, self).__init__()
        args = filters

        # first layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, args[0], kernel_size=3, padding='same'),
            nn.ReLU(), nn.Dropout(p=dropout),
            depthwise_separable_conv(args[0], 1, args[0]), nn.ReLU())

        self.input_skip = nn.Sequential(
            nn.Conv2d(
                channel,
                args[0],
                kernel_size=3,
                stride=2,
            ))
        self.pool = nn.MaxPool2d(2)
        self.down1 = ConvDown(args[0], args[1])
        self.res1 = ResidualDown(args[0], args[1])
        self.down2 = ConvDown(args[1], args[2])
        self.res2 = ResidualDown(args[1], args[2])

        self.bridge = ResidualBridge(args[2], args[3])

    def forward(self, x):
        xpad = F.pad(x, [0, 1, 0, 1], mode='replicate')
        x1 = self.input_layer(x)
        x1b = self.pool(x1) + self.input_skip(xpad)

        x2 = self.down1(x1b)
        x2b = self.pool(x2) + self.res1(x1b)

        x3 = self.down2(x2b)
        x3b = self.pool(x3) + self.res2(x2b)

        x4 = self.bridge(x3b)

        embeddings = [x1, x2, x3, x3b, x4]
        return embeddings


class ResUnetDecoder2(nn.Module):

    def __init__(self, dropout=0.3, filters=None):
        super(ResUnetDecoder2, self).__init__()

        args = filters
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.res_bridge = ResidualUp(args[2], args[3])
        self.up1 = ConvUp(args[3], args[2])
        self.res3 = ResidualUp(args[3], args[2])
        self.up2 = ConvUp(args[2], args[1])
        self.res4 = ResidualUp(args[2], args[1])

        # last layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(args[1] + args[1] // 2,
                      args[0],
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(args[0], args[0], kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(args[0], 1, kernel_size=1, padding='same'),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2, x3, x3b, x4):
        # x ist a list of embeddings from encoder

        x4b = self.upsample(x4) + self.res_bridge(x3b)

        x5 = torch.cat([x4b, x3], dim=1)
        x6 = self.up1(x5) + self.res3(x4b)

        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up2(x7) + self.res4(x6)

        x9 = torch.cat([x8, x1], dim=1)
        output = self.output_layer(x9)

        return output


class UNet_Transfer(pl.LightningModule):

    def __init__(self,
                 lr=0.0002,
                 backbone=nn.Module,
                 dropout=0.3,
                 batch_size=3,
                 finetune=False,
                 seed=42,
                 filters=[32, 64, 128, 256]):
        super().__init__()
        """
        Args:
            backbone: pretrained encoder
            finetune: if false -> don't update parameters of encoder 
                     if true > update all parameters (encoder+ decoder)
        """

        self.model_type = 'UNET_Transfer'
        self.finetune = finetune
        self.filters = filters
        self.dropout = dropout

        # Hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.ce = nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()
        self.save_hyperparameters()

        if backbone == None:
            print('Backbone/head not loaded')
            return

        # layers
        self.encoder = backbone
        self.decoder = ResUnetDecoder2(self.dropout, self.filters)

        if self.finetune == False:
            # freeze params of encoder
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        embedding = self.encoder(x)

        x1 = embedding[0]
        x2 = embedding[1]
        x3 = embedding[2]
        x3b = embedding[3]
        x4 = embedding[4]
        # print('Enocder2:',x1.shape, x2.shape, x3.shape, x3b.shape, x4.shape)

        x = self.decoder(x1, x2, x3, x3b, x4)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        # self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar=True, logger=True)
        # self.logger.experiment.add_scalar('train_loss', loss, global_step=self.global_step)

        y_true = y.detach()
        # y_pred = y_pred.argmax(-1).detach()
        return {'loss': loss, 'y_pred': y_pred, 'y_true': y}

    def training_epoch_end(self, outputs):
        y_true_list = list()
        y_pred_list = list()
        # add here accuracy

    def validation_step(self, val_batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        if self.finetune:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)

        return optimizer

    def test_step(self, test_batch, batch_idx):
        pass

    def test_step_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass


# %%

channels = cfg["UNet"]['channels']
dropout = cfg["UNet"]['dropout']

filters = [32, 64, 128, 256]
_encoder = ResUnetEncoder3(channel=1, filters=filters, dropout=dropout)
model = UNet_Transfer(backbone=_encoder)

# tb_logger = pl_loggers.TensorBoardLogger(save_dir="../../logs/")
trainer = pl.Trainer(
    gpus=cfg["pretraining"]['gpus'],
    deterministic=True,
    max_epochs=cfg["pretraining"]['epochs'])  # , logger=tb_logger

# fit the first time with one augmentation
trainer.fit(model, my_dataloader)

# %%

iterator = iter(my_dataloader)
x, y = next(iterator)
x

# %%

unet = ssl.model.ResUnetEncoder(1, filters=[32, 64, 128, 256])
summary(unet, (1, 256, 256), 10)

# %%

unet = ResUnetEncoder3(1, filters=[32, 64, 128, 256])
summary(unet, (1, 256, 256), 10)

# %%

unet = UNet_Transfer(backbone=ResUnetEncoder3(3, filters=[32, 64, 128, 256]))
summary(unet, (3, 256, 256), 10)

# %%

plt.imshow(dataset_train[0][0].swapaxes(0, 2))
# %%

np.swapaxes(dataset_train[0][0], 0, 3)
# %%
dataset_train[0][0].swapaxes(0, 2).shape
# %%
x_train_unsup.shape
# %%
