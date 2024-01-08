import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from selfsupervised.model.torchnn.positional_encoding import PositionalEncoding


class Max(nn.Module):

    def forward(self, x):
        return x.max(1)[0]


class TransformerEncoder(pl.LightningModule):
    """ Transformer Encoder with classification head
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
    def __init__(self,
                 input_dim=13,
                 seq_length=14,
                 num_classes=7,
                 d_model=64,
                 n_head=2,
                 d_ffn=128,
                 nlayers=2,
                 dropout=0.018,
                 activation="relu",
                 lr=0.0002,
                 batch_size=3,
                 seed=42,
                 PositonalEncoding=False) -> None:
        super().__init__()
        self.model_type = 'Transformer Encoder'
        pl.seed_everything(seed)
        self.PositionalEncoding = PositionalEncoding
        self.seq_length = seq_length

        # Hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.ce = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        # Layers
        encoder_layers = nn.TransformerEncoderLayer(d_model,
                                                    n_head,
                                                    dim_feedforward=d_ffn,
                                                    dropout=dropout,
                                                    activation=activation,
                                                    batch_first=True)
        # positional encoding
        if self.PositionalEncoding:

            self.backbone = nn.Sequential(
                nn.Linear(input_dim, d_model), nn.ReLU(),
                PositionalEncoding(d_model=d_model, dropout=0),
                nn.TransformerEncoder(encoder_layers, nlayers,
                                      nn.LayerNorm(d_model)), Max(), nn.ReLU())
        else:
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, d_model), nn.ReLU(),
                nn.TransformerEncoder(encoder_layers, nlayers,
                                      nn.LayerNorm(d_model)), Max(), nn.ReLU())
        # classification output layer
        self.outlinear = nn.Sequential(nn.Linear(d_model, num_classes))

    def forward(self, x):
        # N x T x D -> N x T x d_model / Batch First!
        embedding = self.backbone(x)
        x = self.outlinear(embedding)
        # torch.Size([N,num_classes ])
        x = F.log_softmax(x, dim=-1)
        # torch.Size([N, num_classes])
        return x, embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred, embedding = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        y_true = y.detach()
        y_pred = y_pred.argmax(-1).detach()
        return {
            'loss': loss,
            'y_pred': y_pred,
            'y_true': y_true,
            'embedding': embedding.detach()
        }

    def training_epoch_end(self, outputs):
        y_true_list = list()
        y_pred_list = list()
        embedding_list = list()

        for item in outputs:
            y_true_list.append(item['y_true'])
            y_pred_list.append(item['y_pred'])
            embedding_list.append(item['embedding'])

        acc = accuracy_score(
            torch.cat(y_true_list).cpu(),
            torch.cat(y_pred_list).cpu())
        # overall accuracy
        self.log('OA', round(acc, 2), logger=True) # type: ignore
        if not self.current_epoch % 10:
            self.logger.experiment.add_embedding(
                torch.cat(embedding_list),
                metadata=torch.cat(y_true_list),
                global_step=self.current_epoch,
                tag='supervised_embedding')

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred, embedding = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('val_loss', loss, logger=True)
        y_true = y.detach()
        y_pred = y_pred.argmax(-1).detach()
        return {'loss': loss, 'y_pred': y_pred, 'y_true': y_true}

    def validation_epoch_end(self, outputs):
        y_true_list = list()
        y_pred_list = list()

        for item in outputs:
            y_true_list.append(item['y_true'])
            y_pred_list.append(item['y_pred'])

        acc = accuracy_score(
            torch.cat(y_true_list).cpu(),
            torch.cat(y_pred_list).cpu())
        # overall accuracy
        self.log('OA', round(acc, 2), logger=True) # type: ignore

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred, embedding = self.forward(x)
        loss = self.ce(y_pred, y)
        # self.log('test_results', loss,on_step=True,prog_bar=True)

        y_true = y.detach()
        y_pred = y_pred.argmax(-1).detach()
        return {'loss': loss, 'y_pred': y_pred, 'y_true': y_true}

    def test_step_end(self, outputs):
        return outputs

    def test_epoch_end(self, outputs):
        # gets all results from test_steps
        y_true_list = list()
        y_pred_list = list()

        for item in outputs:
            y_true_list.append(item['y_true'])
            y_pred_list.append(item['y_pred'])

        acc = accuracy_score(
            torch.cat(y_true_list).cpu(),
            torch.cat(y_pred_list).cpu())
        # Overall accuracy
        self.log('OA', round(acc, 2), logger=True) # type: ignore