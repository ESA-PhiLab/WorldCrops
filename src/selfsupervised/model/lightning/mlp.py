import pytorch_lightning as pl
import torch
import torch.nn as nn


class MLP(pl.LightningModule):
    """ A Multilayer Perceptron with 3 linear layers and Relu activation
        Args:
            input_dim: amount of dimensions (width x height x channels)
            num_classes: amount of target classes
    """

    def __init__(self, input_dim, num_classes, lr=1e-4) -> None:
        super().__init__()
        self.input_dim: int = input_dim
        self.num_classes: int = num_classes

        self.model_type: str = 'MLP based on Pytorch Lightning'
        self.lr: float = lr
        self.layers = nn.Sequential(nn.Linear(self.input_dim, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU(),
                                    nn.Linear(32, self.num_classes))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch):
        x, y = batch
        x = x.flatten(1)
        y_pred = self.layers(x)
        loss = self.ce(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def validation_step(self, val_batch):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.log('val_loss', loss)
