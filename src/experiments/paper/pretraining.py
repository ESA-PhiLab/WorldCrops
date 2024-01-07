# %%
"""
    Example of pre-training and transfer of a model for crop types 
    to reproduce the results in the paper: https://arxiv.org/abs/2204.02100
"""

import copy
import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml

# import selfsupervised
from selfsupervised.data.croptypes.datamodules import (AugmentationExperiments,
                                                       BavariaDataModule)
from selfsupervised.model.lightning.simsiam import SimSiam
from selfsupervised.model.lightning.transformer_transfer import \
    TransformerTransfer
from selfsupervised.processing.utils import (clean_bavarian_labels,
                                             remove_false_observation_RF,
                                             seed_torch)

# %%
################################################################
# Configuration
################################################################
# IARAI / ESA
IARAI = False

if os.path.isfile(sys.argv[1]) and os.access(sys.argv[1], os.R_OK):
    # Open both config files as dicts and combine them into a single dict.
    with open(sys.argv[1], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
else:
    with open("../../../config/croptypes/param_pretraining.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


# training parameter
num_workers = cfg["pretraining"]['num_workers']
epochs_pretraining = cfg["pretraining"]['epochs']
device = cfg["pretraining"]['device']
lr = cfg["pretraining"]['learning_rate']
# positional encoding
pe = cfg["pretraining"]['pe']
no_gpus = cfg["pretraining"]['no_gpus']

epochs_fine = cfg["finetuning"]['epochs']
batch_size = cfg["finetuning"]['batch_size']

# transformer
input_dim = cfg["transformer"]['input_dim']
num_classes = cfg["transformer"]['num_classes']
n_head = cfg["transformer"]['n_head']
nlayers = cfg["transformer"]['nlayers']

# definitions for simsiam
num_ftrs = cfg["pretraining"]['num_ftrs']
proj_hidden_dim = cfg["pretraining"]['proj_hidden_dim']
pred_hidden_dim = cfg["pretraining"]['pred_hidden_dim']
out_dim = cfg["pretraining"]['out_dim']
batch_size_sim = cfg["pretraining"]['batch_size']

# PATH to data files
training_data = '../../../data/cropdata/Bavaria/sentinel-2/Training_bavaria.xlsx'
trained_mdoels = '../../../logs/trained_models/'
################################
seed_torch()


# %%
# load data for bavaria
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

# SimSiam augmentation data
# augmentation between years independent of crop type
dm_years = AugmentationExperiments(data_dir=training_data,
                                   batch_size=batch_size_sim, num_workers=num_workers, experiment='Experiment4')

# data for invariance between crops
dm_crops1 = AugmentationExperiments(data_dir=training_data,
                                    batch_size=batch_size_sim, num_workers=num_workers, experiment='Experiment3')
dm_crops2 = AugmentationExperiments(data_dir=training_data,
                                    batch_size=batch_size_sim, num_workers=num_workers, experiment='Experiment5')
dm_crops3 = AugmentationExperiments(data_dir=training_data,
                                    batch_size=batch_size_sim, num_workers=num_workers, experiment='Experiment6')
dm_crops4 = AugmentationExperiments(data_dir=training_data,
                                    batch_size=batch_size_sim, num_workers=num_workers, experiment='Experiment7')
# %%
# Vorgehen:
# 1. Pre-Train transformer unsupervised 
# 2. Finetune

# %%
# call transformer encoder definition based on torch.nn
from selfsupervised.model.torchnn.transformer_enc_seq import \
    TransformerEncoder as Attention

transformer = Attention(input_dim=input_dim,
                        num_classes=num_classes, n_head=n_head, nlayers=nlayers)

backbone = nn.Sequential(*list(transformer.children())[-2])
model_sim = SimSiam(backbone, num_ftrs=num_ftrs, proj_hidden_dim=proj_hidden_dim,
                       pred_hidden_dim=pred_hidden_dim, out_dim=out_dim, lr=lr)


if IARAI:
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp',
                         deterministic=True, max_epochs=epochs_pretraining)
else:
    trainer = pl.Trainer(deterministic=True, max_epochs=epochs_pretraining)

# fit the first time with one augmentation
trainer.fit(model_sim, datamodule=dm_crops1)
torch.save(backbone, trained_mdoels + "pretraining1.ckpt")
# %%

transformer2 = Attention(input_dim=input_dim, num_classes=num_classes, n_head=n_head, nlayers=nlayers)
backbone2 = nn.Sequential(*list(transformer2.children())[-2])
model_sim2 = SimSiam(backbone2, num_ftrs=num_ftrs, proj_hidden_dim=proj_hidden_dim,
                        pred_hidden_dim=pred_hidden_dim, out_dim=out_dim, lr=lr)

if IARAI:
    trainer2 = pl.Trainer(gpus=no_gpus, strategy='ddp',
                          deterministic=True, max_epochs=epochs_pretraining)
else:
    trainer2 = pl.Trainer(deterministic=True, max_epochs=epochs_pretraining)
trainer2.fit(model_sim2, datamodule=dm_crops2)
torch.save(backbone2, trained_mdoels + "pretraining2.ckpt")

# %%
transformer3 = Attention(input_dim=input_dim,
                         num_classes=num_classes, n_head=n_head, nlayers=nlayers)
backbone3 = nn.Sequential(*list(transformer3.children())[-2])
model_sim3 = SimSiam(backbone3, num_ftrs=num_ftrs, proj_hidden_dim=proj_hidden_dim,
                        pred_hidden_dim=pred_hidden_dim, out_dim=out_dim, lr=lr)

# fit for invariance between same crops
if IARAI:
    trainer3 = pl.Trainer(gpus=no_gpus, strategy='ddp',
                          deterministic=True, max_epochs=epochs_pretraining)
else:
    trainer3 = pl.Trainer(deterministic=True, max_epochs=epochs_pretraining)

trainer3.fit(model_sim3, datamodule=dm_crops3)
torch.save(backbone3, trained_mdoels + "pretraining3.ckpt")

transformer4 = Attention(input_dim=input_dim,
                         num_classes=num_classes, n_head=n_head, nlayers=nlayers)
backbone4 = nn.Sequential(*list(transformer4.children())[-2])
model_sim4 = SimSiam(backbone4, num_ftrs=num_ftrs, proj_hidden_dim=proj_hidden_dim,
                        pred_hidden_dim=pred_hidden_dim, out_dim=out_dim, lr=lr)

# fit for invariance between same crops
if IARAI:
    trainer4 = pl.Trainer(gpus=no_gpus, strategy='ddp',
                          deterministic=True, max_epochs=epochs_pretraining)
else:
    trainer4 = pl.Trainer(deterministic=True, max_epochs=epochs_pretraining)

trainer4.fit(model_sim4, datamodule=dm_crops4)
torch.save(backbone4, trained_mdoels + "pretraining4.ckpt")
# %%

# %%
# backbone = torch.load("../model/pretrained/backbone_3_aug_17.2.ckpt")
# copy pretrained backbone for experiments
backbone_copy1 = copy.deepcopy(backbone)
backbone_copy2 = copy.deepcopy(backbone2)
backbone_copy3 = copy.deepcopy(backbone3)
backbone_copy4 = copy.deepcopy(backbone4)

# %%
# use pretrained backbone and finetune with a new head
# TransformerTransfer -> finetune = False: only head will be finetuned

from selfsupervised.model.torchnn.head import ThreeLayerHead as Head

# head = nn.Sequential(*list(transformer1.children())[-1])
head = Head(64, 6)


transfer_model = TransformerTransfer(input_dim=input_dim, num_classes=num_classes, d_model=num_ftrs,
                                    backbone=backbone_copy1, head=head, batch_size=batch_size, finetune=False, lr=lr)
if IARAI:
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp',
                         deterministic=True, max_epochs=epochs_fine)
else:
    trainer = pl.Trainer(deterministic=True, max_epochs=epochs_fine)

trainer.fit(transfer_model, datamodule=dm_bavaria)
trainer.test(transfer_model, datamodule=dm_bavaria)

# %%
# head2 = nn.Sequential(*list(transformer2.children())[-1])
head2 = Head(64, 6)


# use pretrained backbone and finetune
transfer_model2 = TransformerTransfer(input_dim=input_dim, num_classes=num_classes, d_model=num_ftrs,
                                     backbone=backbone_copy2, head=head2, batch_size=batch_size, finetune=False, lr=lr)
if IARAI:
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp',
                         deterministic=True, max_epochs=epochs_fine)
else:
    trainer = pl.Trainer(deterministic=True, max_epochs=epochs_fine)

trainer.fit(transfer_model2, datamodule=dm_bavaria2)
trainer.test(transfer_model2, datamodule=dm_bavaria2)
# %%
# head3 = nn.Sequential(*list(transformer3.children())[-1])
head3 = Head(64, 6)

transfer_model3 = TransformerTransfer(input_dim=input_dim, num_classes=num_classes, d_model=num_ftrs,
                                     backbone=backbone_copy3, head=head3, batch_size=batch_size, finetune=False, lr=lr)

if IARAI:
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp',
                         deterministic=True, max_epochs=epochs_fine)
else:
    trainer = pl.Trainer(deterministic=True, max_epochs=epochs_fine)

trainer.fit(transfer_model3, datamodule=dm_bavaria3)
trainer.test(transfer_model3, datamodule=dm_bavaria3)

# %%
# head4 = nn.Sequential(*list(transformer3.children())[-1])
head4 = Head(64, 6)

transfer_model4 = TransformerTransfer(input_dim=input_dim, num_classes=num_classes, d_model=num_ftrs,
                                     backbone=backbone_copy4, head=head4, batch_size=batch_size, finetune=False, lr=lr)

if IARAI:
    trainer = pl.Trainer(gpus=no_gpus, strategy='ddp',
                         deterministic=True, max_epochs=epochs_fine)
else:
    trainer = pl.Trainer(deterministic=True, max_epochs=epochs_fine)

trainer.fit(transfer_model4, datamodule=dm_bavaria4)
trainer.test(transfer_model4, datamodule=dm_bavaria4)

# %%
