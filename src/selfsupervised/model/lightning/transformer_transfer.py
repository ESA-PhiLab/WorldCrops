import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


class TransformerTransfer(pl.LightningModule):
    """ Use a pre-trained transformer encoder and finetune it
        Args:
            input_dim: amount of input dimensions -> Sentinel2 has 13 bands
            num_classes: amount of target classes
            d_model: default = 64 #number of expected features
            backbone: pretrained encoder
            finetune: if false -> don't update parameters of backbone
                     if true > update all parameters (backbone + new head)
    """

    def __init__(self,
                 lr=0.0002,
                 input_dim=13,
                 num_classes=7,
                 d_model=64,
                 backbone=None,
                 head=None,
                 batch_size=3,
                 finetune=False,
                 seed=42) -> None:
        super().__init__()
        self.model_type = 'Transfer Learning'
        self.finetune = finetune

        # Hyperparameters
        self.lr: float = lr
        self.batch_size: int = batch_size
        self.ce = nn.CrossEntropyLoss()
        self.save_hyperparameters()

        # layers
        self.backbone = backbone
        self.outlinear = head

        if (backbone and head) is None:
            print('Backbone/head not loaded')
            return

        if self.finetune is False:
            # freeze params of backbone
            for param in self.backbone.parameters():
                param.requires_grad = False

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
        self.logger.experiment.add_scalar('train_loss',
                                          loss,
                                          global_step=self.global_step)

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
        # self.log('OA',round(acc,2), on_epoch = True, logger=True)
        self.logger.experiment.add_scalar('OA',
                                          round(acc, 2),
                                          global_step=self.current_epoch)

        if not self.current_epoch % 10:
            self.logger.experiment.add_embedding(
                torch.cat(embedding_list),
                metadata=torch.cat(y_true_list),
                global_step=self.current_epoch,
                tag='Finetune_embedding')

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred, embedding = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('val_loss', loss, on_epoch=True, logger=True)
        # self.logger.experiment.add_scalar('val_loss', loss, global_step=self.current_epoch)

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
        self.log('OA', round(acc, 2), on_epoch=True,
                 logger=True)  # type: ignore
        # self.logger.experiment.add_scalar('OA',round(acc,2), global_step=self.current_epoch)

    def configure_optimizers(self):
        if self.finetune:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.outlinear.parameters(),
                                         lr=self.lr)

        return optimizer

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred, embedding = self.forward(x)
        loss = self.ce(y_pred, y)

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
        self.log('OA', round(acc, 2), on_epoch=True,
                 logger=True)  # type: ignore
