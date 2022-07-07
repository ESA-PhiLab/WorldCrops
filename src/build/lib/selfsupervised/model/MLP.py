######################
# MLP 
######################

import pytorch_lightning as pl
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        """
        Args:
            input_dim: amount of dimensions (width x height x channels)
            num_classes: amount of target classes
            ....
        Input:
            batch size x (width x height x channels)
        """

        self.input_dim = input_dim
        self.num_target_classes = num_classes

        self.layers = nn.Sequential(
            # convert input dim (width, height and channels)
            nn.Flatten(),
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_target_classes)
        )

    def forward(self, x):
        return self.layers(x)


class MLP_LM(pl.LightningModule):
  
  def __init__(self, input_dim, num_classes, lr = 1e-4):
    super().__init__()
    """
    Args:
        input_dim: amount of dimensions (width x height x channels)
        num_classes: amount of target classes
        ....
    Input:
        batch size x (width x height x channels)
    """
    

    self.model_type = 'MLP_LM'
    self.input_dim = input_dim
    self.num_classes = num_classes

    self.layers = nn.Sequential(
      nn.Linear(self.input_dim, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, self.num_classes)
    )
    self.ce = nn.CrossEntropyLoss()
    
  def forward(self, x):
    return self.layers(x)
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    x = x.flatten(1)
    y_pred = self.layers(x)
    loss = self.ce(y_pred, y)
    self.log('train_loss', loss)
    return loss
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    return optimizer

  def validation_step(self, val_batch, batch_idx):
      x, y = val_batch
      y_pred = self.forward(x)
      loss = self.ce(y_pred, y)
      self.log('val_loss', loss)