##########################
# Attention Transformer 
##########################

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

class Attention(nn.Module):
  
    def __init__(self, input_dim = 13, num_classes = 7, d_model = 64, n_head = 2, d_ffn = 128, nlayers = 2, dropout = 0.018, activation="relu"):
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
        """

        self.model_type = 'Transformer'
        

        self.inlinear = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU()
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout = dropout, activation=activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers, nn.LayerNorm(d_model))
        self.outlinear = nn.Linear(d_model, num_classes)

    def forward(self,x):
        # N x T x D -> N x T x d_model / Batch First!
        x = self.inlinear(x)
        #torch.Size([N, T, d_model]) 
        x = self.relu(x)
        x = self.transformer_encoder(x)
        #torch.Size([N, T, d_model])
        x = x.max(1)[0]
        x = self.relu(x)
        x = self.outlinear(x)
        #torch.Size([N,num_classes ])
        x = F.log_softmax(x, dim=-1)
        #torch.Size([N, num_classes])
        return x

class Max(nn.Module):
    def __init__(self, dim=None, keepdim=False): self.dim, self.keepdim = dim, keepdim
    def forward(self, x): return x.max(self.dim, keepdim=self.keepdim)[0]
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim}, keepdim={self.keepdim})'

class Attention2(nn.Module):
  
    def __init__(self, input_dim = 13, num_classes = 7, d_model = 64, n_head = 2, d_ffn = 128, nlayers = 2, dropout = 0.018, activation="relu"):
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
        """

        self.model_type = 'Transformer'
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout = dropout, activation=activation, batch_first=True)

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.TransformerEncoder(encoder_layers, nlayers, nn.LayerNorm(d_model)),
            Max(1),
            nn.ReLU()
        )
        self.outlinear = nn.Sequential(
            nn.Linear(d_model, num_classes)
        )


    def forward(self,x):
        # N x T x D -> N x T x d_model / Batch First!
        x = self.backbone(x)
        x = self.outlinear(x)
        #torch.Size([N,num_classes ])
        x = F.log_softmax(x, dim=-1)
        #torch.Size([N, num_classes])
        return x


class Attention_LM(pl.LightningModule):

    def __init__(self, input_dim = 13, num_classes = 7, d_model = 64, n_head = 2, d_ffn = 128, nlayers = 2, dropout = 0.018, activation="relu", lr = 0.0002, batch_size  = 3):
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

        # Hyperparameters
        self.lr = lr
        self.batch_size = batch_size
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
        print(x.size())
        x = x.max(1)[0]
        print(x.size())
        x = self.relu(x)
        print(x.size())
        x = self.outlinear(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        y_true = y.detach()
        y_pred = y_pred.argmax(-1).detach()
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y}

    def training_epoch_end(self, outputs):
        y_true_list = list()
        y_pred_list = list()

        for item in outputs:
            y_true_list.append(item['y_true'])
            y_pred_list.append(item['y_pred'])

        acc = accuracy_score(torch.cat(y_true_list),torch.cat(y_pred_list))
        #overall accuracy
        self.log('OA',round(acc,2)) 

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('val_loss', loss)
        y_true = y.detach()
        y_pred = y_pred.argmax(-1).detach()
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y}


    def validation_epoch_end(self, outputs):
        y_true_list = list()
        y_pred_list = list()

        for item in outputs:
            y_true_list.append(item['y_true'])
            y_pred_list.append(item['y_pred'])

        acc = accuracy_score(torch.cat(y_true_list),torch.cat(y_pred_list))
        #overall accuracy
        self.log('OA',round(acc,2))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('test_results', {'test_loss' : loss},on_step=True,prog_bar=True)

        y_true = y.detach()
        y_pred = y_pred.argmax(-1).detach()
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y}

    def test_step_end(self, outputs):
        return outputs

    def test_epoch_end(self, outputs):
        #gets all results from test_steps
        y_true_list = list()
        y_pred_list = list()

        for item in outputs:
            y_true_list.append(item['y_true'])
            y_pred_list.append(item['y_pred'])

        acc = accuracy_score(torch.cat(y_true_list),torch.cat(y_pred_list))
        #Overall accuracy
        self.log('OA',round(acc,2))

class Attention_Transfer(pl.LightningModule):

    def __init__(self, lr = 0.0002, input_dim = 13, num_classes = 7,d_model = 64, backbone=None, batch_size  = 3, transfer = False):
        super().__init__()
        """
        Args:
            input_dim: amount of input dimensions -> Sentinel2 has 13 bands
            num_classes: amount of target classes
            d_model: default = 64 #number of expected features
            backbone: pretrained encoder
            tranfer: if false -> don't update parameters of backbone (only new linear head) 
                     if true > update all parameters (backbone + new head)
        """

        self.model_type = 'Transformer_LM'
        self.transfer = transfer

        # Hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.ce = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        
        # Layers
        self.backbone = backbone
        #self.relu = nn.ReLU()
        self.outlinear = nn.Linear(d_model, num_classes)

        if backbone == None:
            print('Backbone not loaded')
            return

        if self.transfer == False:
            # freeze params
            for param in self.backbone.parameters():
                param.requires_grad = False


        
    def forward(self,x):
        # N x T x D -> N x T x d_model / Batch First!
        x = self.backbone(x)
        #torch.Size([N, T, d_model])
        print(x.size())
        #x = x.max(1)[0]
        #print(x.size())
        #x = self.relu(x)
        x = self.outlinear(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        y_true = y.detach()
        y_pred = y_pred.argmax(-1).detach()
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y}

    def training_epoch_end(self, outputs):
        y_true_list = list()
        y_pred_list = list()

        for item in outputs:
            y_true_list.append(item['y_true'])
            y_pred_list.append(item['y_pred'])

        acc = accuracy_score(torch.cat(y_true_list),torch.cat(y_pred_list))
        #overall accuracy
        self.log('OA',round(acc,2)) 

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('val_loss', loss)
        y_true = y.detach()
        y_pred = y_pred.argmax(-1).detach()
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y}


    def validation_epoch_end(self, outputs):
        y_true_list = list()
        y_pred_list = list()

        for item in outputs:
            y_true_list.append(item['y_true'])
            y_pred_list.append(item['y_pred'])

        acc = accuracy_score(torch.cat(y_true_list),torch.cat(y_pred_list))
        #overall accuracy
        self.log('OA',round(acc,2))

    def configure_optimizers(self):
        if self.transfer:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.outlinear.parameters(),lr = self.lr)
        
        return optimizer

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('test_results', {'test_loss' : loss},on_step=True,prog_bar=True)

        y_true = y.detach()
        y_pred = y_pred.argmax(-1).detach()
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y}

    def test_step_end(self, outputs):
        return outputs

    def test_epoch_end(self, outputs):
        #gets all results from test_steps
        y_true_list = list()
        y_pred_list = list()

        for item in outputs:
            y_true_list.append(item['y_true'])
            y_pred_list.append(item['y_pred'])

        acc = accuracy_score(torch.cat(y_true_list),torch.cat(y_pred_list))
        #Overall accuracy
        self.log('OA',round(acc,2))
        



    
