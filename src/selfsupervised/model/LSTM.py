######################
# LSTM
######################

import pytorch_lightning as pl
import torch
import torch.nn as nn


class LSTM_LM(pl.LightningModule):

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.lr = 1e-4
        self.model_type = 'LSTM-notready'
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
