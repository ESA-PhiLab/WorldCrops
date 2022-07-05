######################
# RNN
######################

import pytorch_lightning as pl
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout = 0.,bidirectional=True):
        super().__init__()
        """
        Args:
            input_dim: amount of input dimensions 
            num_classes: amount of target classes
            + ... : https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        Input:
            batch size(N) x T (sequence) x D (dim)
        """

        self.model_type = 'RNN'
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=2, bidirectional=bidirectional, batch_first=True, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size * (1 + bidirectional) , output_size)

    def forward(self,x):
        #x = x.transpose(2,1)    # [N x D x T] --> [N x T x D]
        output, hidden_ = self.rnn(x) # all T steps [N x T x hidden_size * (1 + bidirectional)]
        output = output[:, -1]  # take last: [N x hidden_size * (1 + bidirectional)]
        output = self.linear(self.dropout(output))
        return output


class RNN_LM(pl.LightningModule):

    def __init__(self, input_size, output_size, hidden_size, dropout = 0.,bidirectional=True):
        super().__init__()
        """
        Args:
            input_dim: amount of input dimensions 
            num_classes: amount of target classes
            ....
        Input:
            batch size(N) x T (sequence) x D (dim)
        """

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=2, bidirectional=bidirectional, batch_first=True, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size * (1 + bidirectional) , output_size)

        self.model_type = 'RNN_LM'
        self.lr = 0.0002
        self.ce = nn.CrossEntropyLoss()

    def forward(self,x):
        #x = x.transpose(2,1)    # [N x D x T] --> [N x T x D]
        output, hidden_ = self.rnn(x) # all T steps [N x T x hidden_size * (1 + bidirectional)]
        output = output[:, -1]  # take last: [N x hidden_size * (1 + bidirectional)]
        output = self.linear(self.dropout(output))
        return output

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