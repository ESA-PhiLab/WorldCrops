import pytorch_lightning as pl
import torch
import torch.nn as nn


class RNN(pl.LightningModule):
    """ simple RNN
        Args:
            input_size: input size
            hidden_size: hidden dim
            output_size: output size
    """

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 dropout=0.,
                 bidirectional=True, lr=0.0002) -> None:
        super().__init__()
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.hidden_size: int = hidden_size
        self.model_type: str = 'RNN based on Pytorch Lightning'
        self.lr: float = lr
        self.rnn = nn.RNN(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=2,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.hidden_size * (1 + bidirectional), self.output_size)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        # x = x.transpose(2,1)    # [N x D x T] --> [N x T x D]
        output, hidden_ = self.rnn(
            x)  # all T steps [N x T x hidden_size * (1 + bidirectional)]
        output = output[:,
                        -1]  # take last: [N x hidden_size * (1 + bidirectional)]
        output = self.linear(self.dropout(output))
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
