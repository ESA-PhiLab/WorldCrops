#%%
import pytorch_lightning as pl
import importlib
import sys 
import torch
from torch import nn
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# MODEL
sys.path.append(   '/iarai/home/daniel.springer/Projects/InvPro/repo/greens_function/models/hopfield/')
from models import get_dataloader

class lightning_wraper(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        sys.path.append(conf['PATH'])
        _module = importlib.import_module(conf['LOAD_MODEL_TYPE'], package=None)
        _model = getattr(_module, conf['LOAD_MODEL_NAME'])
        self.model = _model(conf)
        data = np.load('Bavaria_13.npy', allow_pickle=True).item()
        self.stored_pattern = data['data']
        self.criterion_mse = nn.MSELoss()
        self.lr = conf['lr']
        self.weight_decay = conf['weight_decay']
        
    def forward(self, in_sample, base):
        return self.model(in_sample, base)

    def training_step(self, data, batch_idx):        
        R = data[0]
        Y = self.stored_pattern
        target = data[1]
        pred = self.forward(R, Y)
        loss = self.criterion_mse(pred.float(), target.float())
        self.log("lm_a/train_loss", loss)
        return loss

    def validation_step(self, data, batch_idx):
        R = data[0]
        Y = self.stored_pattern
        target = data[1]
        pred = self.forward(R, Y)
        loss = self.criterion_mse(pred.float(), target.float())
        self.log("lm_a/val_loss", loss)
        return {"val_loss": loss}
    
    def spectrum(self, data, base):
        return self.forward(R, Y).detach().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

epochs = 600
batch_size = 1
beta = 10
# MODEL_NAME = 'Hopfield_v3_heads'
MODEL_NAME = 'Hopfield_v1_Lookup'
# MODEL_NAME = 'Hopfield_v1_Classifier'

model_conf = {
    'PATH': '/iarai/home/daniel.springer/Projects/InvPro/repo/greens_function/models/hopfield/',
    'LOAD_MODEL_TYPE': 'models',
    'LOAD_MODEL_NAME': MODEL_NAME,
    "emb_dim": 1,
    "hop_heads": 2,
    "hop_layers": 3,
    "hidden_dim": 64,
    "in_dim": 14,
    "out_dim": 13,
    "total_labels": 6,
    "Hopfield_beta": beta,
    "weight_decay": 2e-6,
    "lr": 2e-5
}
model = lightning_wraper(model_conf)
sname = '/iarai/home/daniel.springer/Projects/Hopfield_Crops/Runs_1/'
fname  = str(MODEL_NAME) + '_layer' + str(model_conf['hop_layers']) + '_emb' + str(model_conf['emb_dim']) + '_heads' + str(model_conf['hop_heads']) + '_hiddendim'+ str(model_conf['hidden_dim']) + '_bs' + str(batch_size) + '_beta' + str(beta) + '/'
# checkpoint = torch.load(sname+fname+"version_0/checkpoints/epoch=1450-step=62392.ckpt")
checkpoint = torch.load(sname+fname+"last.ckpt")
model.load_state_dict(checkpoint['state_dict'])

# %%
train_dataloader = get_dataloader(year=[1], batch_size=1)
val_dataloader = get_dataloader(year=[1], batch_size=1)

n = 0
correct = 0
wrong = 0
wrong_list = torch.zeros((6))
for sample in train_dataloader:
    data = sample[0]
    target = sample[1]
    pred = model(data, None)
    # print(pred)
    # print(target)
    if torch.argmax(pred) == torch.argmax(target):
        correct+=1
    else:
        wrong+=1
        wrong_list[torch.argmax(target)] += 1
    if n == -1: break
    n += 1
print(correct/(wrong+correct) )
print(wrong_list)

#%%
print(correct/(wrong+correct) )
print(wrong_list)