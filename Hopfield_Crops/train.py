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

# Lightning module
class lightning_wraper(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        sys.path.append(conf['PATH'])
        _module = importlib.import_module(conf['LOAD_MODEL_TYPE'], package=None)
        _model = getattr(_module, conf['LOAD_MODEL_NAME'])
        self.model = _model(conf)
        data = np.load('Bavaria_13.npy', allow_pickle=True).item()
        self.stored_pattern = data['data']
        self.criterion_mse = nn.CrossEntropyLoss()
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



def torch_train(num_epochs=10, num_gpus=16):
    sname = '/iarai/home/daniel.springer/Projects/Hopfield_Crops/Runs_1/'
    fname  = str(MODEL_NAME) + '_layer' + str(model_conf['hop_layers']) + '_emb' + str(model_conf['emb_dim']) + '_heads' + str(model_conf['hop_heads']) + '_hiddendim'+ str(model_conf['hidden_dim']) + '_outdim'+ str(model_conf['out_dim']) + '_bs' + str(batch_size) + '_beta' + str(beta) + '/'
    print('--------------------------------------')
    print('--------------------------------------')
    print('-------- WORKING PATH ----------------')
    print(sname)
    print(fname)
    print('--------------------------------------')
    print('--------------------------------------')
    # model = lm_200c_1(model_conf)
    model = lightning_wraper(model_conf)
    # print(model)
    # checkpoint = torch.load("/iarai/home/daniel.springer/Projects/InvPro/repo/greens_function/models/hopfield/Version_2_Lightning/Runs_MIT/Hopfield_v8_layer1_emb1_heads1_hiddendim64_bs1_beta10/last_2.ckpt")
    # model.load_state_dict(checkpoint['state_dict'])

    checkpoint_callback = ModelCheckpoint(save_top_k=-1)
    early_stop_callback = EarlyStopping(monitor="lm_a/val_loss", min_delta=0.00, patience=50, verbose=True, mode="min", check_on_train_epoch_end=False)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=TensorBoardLogger(save_dir=sname, name=fname),
        # callbacks=[checkpoint_callback],
        # CAREUL - EARLY STOPPING IS HERE!!!!!!
        callbacks=[checkpoint_callback, early_stop_callback],
        # callbacks=[tune_callback, checkpoint_callback],
        enable_progress_bar=True)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.save_checkpoint(sname + fname + 'last.ckpt')


    print(model_conf)
    print(' This was stored in : ', sname + fname)


#%%

epochs = 2# 6000
batch_size = 1
beta = 10
channels = 13
MODEL_NAME = 'Hopfield_v1_Lookup'
# MODEL_NAME = 'Hopfield_v1_LookupClassifier'
# MODEL_NAME = 'Hopfield_v1_Classifier'
# MODEL_NAME = 'Hopfield_v2_Classifier'

model_conf = {
    'PATH': '/iarai/home/daniel.springer/Projects/InvPro/repo/greens_function/models/hopfield/',
    'LOAD_MODEL_TYPE': 'models',
    'LOAD_MODEL_NAME': MODEL_NAME,
    "channels": channels,
    "emb_dim": 1,
    "hop_heads": 1,
    "hop_layers": 1,
    "hidden_dim": 64,
    "in_dim": 14,
    "out_dim": 64,
    "total_labels": 6,
    "Hopfield_beta": beta,
    "weight_decay": 2e-6,
    "lr": 2e-5
}

train_dataloader = get_dataloader(year=[1], batch_size=batch_size)
val_dataloader = get_dataloader(year=[1], batch_size=batch_size)
'''NOTE: If the validation is a different year, the early stopping is triggered fairly quickly it seems.'''

def main():
    torch_train(num_epochs=epochs, num_gpus=14)

if __name__ == '__main__':
    main()
 

# %%
