import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from selfsupervised.model.torchnn.unet import ResUnetDecoder


class UNetTransfer(pl.LightningModule):
    """ Unet Transfer Learning. Finetune Unet encoder and decoder 
    Args:
        backbone: pretrained encoder
        finetune: if false -> don't update parameters of encoder 
                    if true > update all parameters (encoder+ decoder)
    """

    def __init__(self,
                 lr=0.0002,
                 backbone=None,
                 dropout=0.3,
                 filters=[32, 64, 128, 256],
                 batch_size=3,
                 finetune=False,
                 seed=42) -> None:
        super().__init__()
        self.model_type: str = 'UNET_Transfer'
        self.finetune: bool = finetune

        # Hyperparameters
        self.lr: float = lr
        self.batch_size: int = batch_size
        self.ce = nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()
        self.save_hyperparameters()

        if backbone is None:
            print('Backbone/head not loaded')
            return

        # layers
        self.encoder = backbone
        self.decoder = ResUnetDecoder(filters=filters, dropout=dropout)

        if self.finetune is False:
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
        x = self.decoder(x1, x2, x3, x3b, x4)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        # self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar=True, logger=True)
        self.logger.experiment.add_scalar('train_loss',
                                          loss,
                                          global_step=self.global_step)

        y_true = y.detach()
        # y_pred = y_pred.argmax(-1).detach()
        return {'loss': loss, 'y_pred': y_pred, 'y_true': y_true}

    def training_epoch_end(self, outputs):
        # avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # logging using tensorboard logger
        # self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        return outputs

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.logger.experiment.add_scalar('val_loss',
                                          loss,
                                          global_step=self.global_step)
        # y_true = y.detach()
        return {'val_loss': loss}

    def validation_step_end(self, outputs):
        return outputs

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_loss = float(avg_loss)
        dict_ = {'val_loss': avg_loss}
        # print(dict_)
        return dict_

    def configure_optimizers(self):
        if self.finetune:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)

        return optimizer

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        # self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar=True, logger=True)
        self.logger.experiment.add_scalar('test_loss',  # type: ignore
                                          loss,
                                          global_step=self.global_step)
        y_true = y.detach()
        return {'test_loss': loss, 'y_pred': y_pred, 'y_true': y_true}

    def test_step_end(self, outputs):
        return outputs

    def test_epoch_end(self, outputs):
        # print(len(y_pred_list))
        # acc = accuracy_score(torch.cat(y_true_list).cpu(),torch.cat(y_pred_list).cpu())
        # Overall accuracy
        # self.log('OA',round(acc,2), logger=True)
        y_true_list = list()
        y_pred_list = list()

        for item in outputs:
            y_true_list.append(item['y_true'])
            y_pred_list.append(item['y_pred'])

    def evaluate_performance(y_true, y_pred):

        pred_areas = []
        true_areas = []
        for i in range(len(y_true)):
            prediction = y_pred[i, :, :]
            berg_samples = prediction[y_true[i] == 1]
            background_samples = prediction[y_true[i] == 0]
            TP = sum(berg_samples)
            FP = sum(background_samples)
            FN = len(berg_samples) - sum(berg_samples)
            TN = len(background_samples) - sum(background_samples)

            pred_areas.append(sum(y_pred[i, :, :].flatten()))
            true_areas.append(sum(y_true[i].flatten()))

        true_areas = np.array(true_areas)
        pred_areas = np.array(pred_areas)

        flat_pred = y_pred.flatten()
        val_arr = np.concatenate(y_true, axis=0)
        flat_true = val_arr.flatten()

        berg_samples = flat_pred[flat_true == 1]
        background_samples = flat_pred[flat_true == 0]

        TP = sum(berg_samples)
        FP = sum(background_samples)
        FN = len(berg_samples) - sum(berg_samples)
        TN = len(background_samples) - sum(background_samples)

        # dice
        dice = 2 * TP / (2 * TP + FP + FN)

        print('overall accuracy')
        print((TP + TN) / (TP + TN + FP + FN) * 100)
        print('false pos rate')
        print(FP / (TN + FP) * 100)
        print('false neg rate')
        print(FN / (TP + FN) * 100)
        print('area deviations')
        print((pred_areas - true_areas) / true_areas * 100)
        print('abs mean error in area')
        print(np.mean(abs((pred_areas - true_areas) / true_areas)) * 100)
        print('area bias')
        print(np.mean((pred_areas - true_areas) / true_areas) * 100)
        print('f1')
        print(dice)
