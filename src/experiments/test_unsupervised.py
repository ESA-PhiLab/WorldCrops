# %%
# compare the crop type classification of RF and SimSiam
import glob
import math
import re

import h5py
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
from pytorch_lightning import loggers as pl_loggers

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

# %%
# read data from h5

x_train = []
formated_img = []
y_train = []
train_mask = []

S1_fn = []
GT_fn = []

for filename in sorted(glob.glob('../../data/Anne/S1_train_small/S1*.h5')):

    with h5py.File(filename, "r") as f:
        data = np.array(f['/DS1'])
        im = np.transpose(data)
        x_train.append(im)
        S1_fn.append(filename)
        filename = filename[16:]
        # save to temp directory
        formatted = (im * 255 / np.max(im)).astype('uint8')
        data = Image.fromarray(formatted)
        formated_img.append(data)
        data.save('../../data/Anne/tmp/' + filename.split("/", 1)[1] + '.png')

for filename in sorted(glob.glob('../../data/Anne/S1_val_small/S1*.h5')):

    with h5py.File(filename, "r") as f:
        data = np.array(f['/DS1'])
        im = np.transpose(data)
        x_train.append(im)
        S1_fn.append(filename)
        filename = filename[16:]
        # save to temp directory
        formatted = (im * 255 / np.max(im)).astype('uint8')
        data = Image.fromarray(formatted)
        formated_img.append(data)
        data.save('../../data/Anne/tmp2/' + filename.split("/", 1)[1] + '.png')

# %%
# show some examples
fig = plt.figure(figsize=(8, 8))
columns = 8
rows = 8

for i in range(1, columns * rows + 1):
    img = formated_img[i]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

# %%

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

# dataset for training
dataset_train = lightly.data.LightlyDataset(input_dir='../../data/Anne/tmp',
                                            # transform=test_transforms
                                            )

# lightly dataset for embedding
dataset_test = lightly.data.LightlyDataset(input_dir='../../data/Anne/tmp2',
                                           transform=test_transforms)

# create a dataloader for training
dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=10,
                                               shuffle=True,
                                               collate_fn=collate_fn,
                                               drop_last=False,
                                               num_workers=num_workers)

# create a dataloader for embedding
dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=batch_size_pre,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=num_workers)
# %%

# %%
resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = selfsupervised.model.lightning.SimSiam_Images(backbone, num_ftrs, proj_hidden_dim,
                                 pred_hidden_dim, out_dim)


tb_logger = pl_loggers.TensorBoardLogger(save_dir="../../logs/")
trainer = pl.Trainer(gpus=cfg["pretraining"]['gpus'],
                     deterministic=True,
                     max_epochs=cfg["pretraining"]['epochs'],
                     logger=tb_logger)

# fit the first time with one augmentation
# trainer.fit(model, dataloader_train)


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


class SimSiam_UNet_Encoder2(pl.LightningModule):

    def __init__(self,
                 backbone=nn.Module,
                 num_ftrs=64,
                 proj_hidden_dim=14,
                 pred_hidden_dim=14,
                 out_dim=14,
                 lr=0.02,
                 weight_decay=5e-4,
                 momentum=0.9,
                 epochs=10,
                 label=False):
        super().__init__()

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.labels = label

        self.ce = lightly.loss.NegativeCosineSimilarity()
        self.backbone = backbone
        self.model_type = 'SimSiam_LM'
        self.projection = lightly.models.modules.heads.ProjectionHead([
            (num_ftrs * num_ftrs, proj_hidden_dim,
             nn.BatchNorm1d(proj_hidden_dim), nn.ReLU()),
            (proj_hidden_dim, out_dim, nn.BatchNorm1d(out_dim), None)
        ])
        self.prediction = lightly.models.modules.heads.SimSiamPredictionHead(
            out_dim, pred_hidden_dim, out_dim)

        # parameters for logging
        self.avg_loss = 0.
        self.avg_output_std = 0.
        self.collapse_level = 0.
        self.out_dim = out_dim

    def forward(self, x0, x1):
        # print("Input Sim1:",x0.shape)
        f0 = self.backbone(x0)
        f0 = f0[4].flatten(start_dim=1)
        # print("After backbone Sim2:",f0.shape)
        f1 = self.backbone(x1)
        f1 = f1[4].flatten(start_dim=1)

        z0 = self.projection(f0)
        z1 = self.projection(f1)

        p0 = self.prediction(z0)
        p1 = self.prediction(z1)

        z0 = z0.detach()
        z1 = z1.detach()
        return (z0, p0), (z1, p1), f0

    def training_step(self, batch, batch_idx):
        (x0, x1), labels, names = batch

        if self.labels:
            (z0, p0), (z1, p1), embedding = self.forward(x0, x1)
            loss = 0.5 * (self.ce(z0, p1) + self.ce(z1, p0))

            # collapse based on https://docs.lightly.ai/tutorials/package/tutorial_simsiam_esa.html
            output = p0.detach()
            output = torch.nn.functional.normalize(output, dim=1)
            output_std = torch.std(output, 0)
            output_std = output_std.mean()

            # use moving averages to track the loss and standard deviation
            w = 0.9
            self.avg_loss = w * self.avg_loss + (1 - w) * loss.item()
            self.avg_output_std = w * self.avg_output_std + (
                1 - w) * output_std.item()

            self.logger.experiment.add_scalar('train_loss_ssl',
                                              loss,
                                              global_step=self.current_epoch)
            self.logger.experiment.add_scalar('Avgloss',
                                              self.avg_loss,
                                              global_step=self.current_epoch)
            self.logger.experiment.add_scalar('Avgstd',
                                              self.avg_output_std,
                                              global_step=self.current_epoch)
            return {
                'loss': loss,
                'y_true': labels.detach(),
                'embedding': embedding.detach()
            }

        else:
            (z0, p0), (z1, p1), embedding = self.forward(x0, x1)
            loss = 0.5 * (self.ce(z0, p1) + self.ce(z1, p0))

            # collapse based on https://docs.lightly.ai/tutorials/package/tutorial_simsiam_esa.html
            output = p0.detach()
            output = torch.nn.functional.normalize(output, dim=1)
            output_std = torch.std(output, 0)
            output_std = output_std.mean()

            # use moving averages to track the loss and standard deviation
            w = 0.9
            self.avg_loss = w * self.avg_loss + (1 - w) * loss.item()
            self.avg_output_std = w * self.avg_output_std + (
                1 - w) * output_std.item()

            self.logger.experiment.add_scalar('train_loss_ssl',
                                              loss,
                                              global_step=self.current_epoch)
            self.logger.experiment.add_scalar('Avgloss',
                                              self.avg_loss,
                                              global_step=self.current_epoch)
            self.logger.experiment.add_scalar('Avgstd',
                                              self.avg_output_std,
                                              global_step=self.current_epoch)
            return {'loss': loss, 'embedding': embedding.detach()}

    def training_epoch_end(self, outputs):
        # the level of collapse is large if the standard deviation of the l2
        # normalized output is much smaller than 1 / sqrt(dim)
        self.collapse_level = max(
            0., 1 - math.sqrt(self.out_dim) * self.avg_output_std)
        # self.log('Collapse Level', round(self.collapse_level,2), logger=True)
        self.logger.experiment.add_scalar('Collapse Level',
                                          round(self.collapse_level, 2),
                                          global_step=self.current_epoch)

        embedding_list = list()
        y_true_list = list()

        if self.labels:
            for item in outputs:
                embedding_list.append(item['embedding'])
                y_true_list.append(item['y_true'])

            # log every 10 epochs
            if not self.current_epoch % 10:
                self.logger.experiment.add_embedding(
                    torch.cat(embedding_list),
                    metadata=torch.cat(y_true_list),
                    global_step=self.current_epoch,
                    tag='pretraining embedding')

        else:
            for item in outputs:
                embedding_list.append(item['embedding'])
            # log every 10 epochs
            if not self.current_epoch % 10:
                self.logger.experiment.add_embedding(
                    torch.cat(embedding_list),
                    global_step=self.current_epoch,
                    tag='pretraining embedding')

    def validation_step(self, val_batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.epochs)
        return [optimizer], [scheduler]


# %%
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
    # slight random cropping
    min_scale=0.5,
    # use a weak color jitter for invariance w.r.t small color changes
    cj_prob=0.2,
    cj_bright=0.1,
    cj_contrast=0.1,
    cj_hue=0.1,
    cj_sat=0.1,
    # gaussian blur to be invariant for texture details
    gaussian_blur=0.3)

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
])

dataset_train = lightly.data.LightlyDataset(input_dir=path_to_data +
                                            'PNG_train/',
                                            # transform=test_transforms
                                            )
dataset_unlabeled = lightly.data.LightlyDataset(input_dir=path_to_data +
                                                'PNG_unlabeled/',
                                                # transform=test_transforms
                                                )

# create a lightly dataset for embedding
dataset_test = lightly.data.LightlyDataset(
    input_dir='../../data/Anne/PNG_test', transform=test_transforms)

train_dev = torch.utils.data.ConcatDataset([dataset_train, dataset_unlabeled])

dataloader_train_unsupervised = torch.utils.data.DataLoader(
    train_dev,
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
# let's get all jpg filenames from a folder
glob_to_data = path_to_data + 'PNG_unlabeled/S1*.png'
fnames = glob.glob(glob_to_data)

# load the first two images using pillow
input_images = [Image.open(fname) for fname in fnames[:2]]

# plot the images
fig = lightly.utils.debug.plot_augmented_images(input_images, collate_fn)

# %%
filters = [32, 64, 128, 256]
channels = cfg["UNet"]['channels']
dropout = cfg["UNet"]['dropout']

_encoder = ResUnetEncoder3(channel=3, filters=filters, dropout=dropout)
model = SimSiam_UNet_Encoder2(_encoder, num_ftrs, proj_hidden_dim,
                              pred_hidden_dim, out_dim)

# tb_logger = pl_loggers.TensorBoardLogger(save_dir="../../logs/")
trainer = pl.Trainer(gpus=cfg["pretraining"]['gpus'],
                     deterministic=True,
                     max_epochs=cfg["pretraining"]['epochs'])

# fit the first time with one augmentation
trainer.fit(model, dataloader_train_unsupervised)
# %%

iterator = iter(dataloader_train_unsupervised)
inputs = next(iterator)
inputs
# %%
print(np.shape(x_train_unsup))

# %%

len(dataset_train)
# %%
