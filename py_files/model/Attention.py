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


class Attention_LM(pl.LightningModule):

    def __init__(self, input_dim = 13, num_classes = 7, d_model = 64, n_head = 2, d_ffn = 128, nlayers = 2, dropout = 0.018, activation="relu", lr = 0.0002):
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
        x = x.max(1)[0]
        x = self.relu(x)
        x = self.outlinear(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('test_loss', loss)
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y, 'y_pred_idx': y_pred.argmax(-1), 'y_score': y_pred.exp()}


    def test_step_end(self, test_outputs):
        accuracy = list()
        y_true_list = list()
        y_pred_list = list()
        y_score_list = list()
        losses = list()

        for out in test_outputs:
            print(out['y_true'], type(out['y_true']))
            accuracy.append(accuracy_score(out['y_true'],out['y_pred']))
            y_true_list.append(out['y_true'])
            #y_pred_list.append(logprobabilities.argmax(-1))
            #y_score_list.append(logprobabilities.exp())


        accuracy = torch.mean(torch.stack(accuracy))
        print(f"Test Accuracy: {round(accuracy,2)}")

        return torch.cat(y_true_list)


        
