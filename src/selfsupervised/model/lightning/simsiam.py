import math

import lightly  # ensure lightly is installed
import pytorch_lightning as pl
import torch
import torch.nn as nn
from lightly.models.modules.heads import ProjectionHead, SimSiamPredictionHead


class SimSiam(pl.LightningModule):
    """
    An implementation of the SimSiam architecture using PyTorch Lightning. SimSiam
    is a self-supervised learning method for learning visual representations without
    labeled data. It operates by maximizing the similarity between two different
    augmented views of the same image or time series.

    Attributes:
        lr (float): Learning rate for the optimizer.
        momentum (float): Momentum factor for SGD optimizer.
        weight_decay (float): Weight decay for regularization in the optimizer.
        epochs (int): Number of training epochs.
        avg_loss (float): A moving average of the training loss.
        avg_output_std (float): A moving average of the standard deviation of the
                                model's outputs, used for monitoring collapse.
        collapse_level (float): A metric to measure the degree of representation
                                collapse in the model.
        out_dim (int): The dimensionality of the output feature vector.
        loss (lightly.loss.NegativeCosineSimilarity): Loss function based on negative
                                                    cosine similarity.
        backbone (nn.Module): Backbone neural network model for feature extraction.
        projection (lightly.models.modules.heads.ProjectionHead): Projection head 
                      that maps the features to a lower-dimensional space.
        prediction (SimSiamPredictionHead): Prediction head of the model that 
                    predicts the representation of one view from the other.

    Args:
        backbone (nn.Module): Backbone model for feature extraction. 
        num_ftrs (int): Number of features in the input tensor expected by the
                        projection head.
        proj_hidden_dim (int): Hidden layer dimension in the projection head.
        pred_hidden_dim (int): Hidden layer dimension in the prediction head.
        out_dim (int): Output dimension for both the projection and prediction heads.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay factor for regularization.
        momentum (float): Momentum factor for the optimizer.
        epochs (int): Number of epochs (T_max) for CosineAnnealingLR (learning rate will be annealed).
    """
    def __init__(self,
                 backbone=nn.Module,
                 num_ftrs=64,
                 proj_hidden_dim=14,
                 pred_hidden_dim=14,
                 out_dim=14,
                 lr=0.02,
                 weight_decay=5e-4,
                 momentum=0.9,
                 epochs=10) -> None:
        super().__init__()
        self.lr: float = lr
        self.momentum: float = momentum
        self.weight_decay: float = weight_decay
        self.epochs: int = epochs
        # parameters for logging
        self.avg_loss: float = 0.
        self.avg_output_std: float = 0.
        self.collapse_level: float = 0.
        self.out_dim: int = out_dim

        self.loss = lightly.loss.NegativeCosineSimilarity()  # type: ignore
        self.backbone = backbone
        self.model_type: str = 'SimSiam based on Pytorch Lightning'
        self.projection = ProjectionHead([
            (num_ftrs, proj_hidden_dim, nn.BatchNorm1d(proj_hidden_dim),
             nn.ReLU()),
            (proj_hidden_dim, out_dim, nn.BatchNorm1d(out_dim), None)
        ])
        self.prediction = SimSiamPredictionHead(out_dim, pred_hidden_dim,
                                                out_dim)
            
    def forward(self, x0, x1):
        f0 = self.backbone(x0)
        f1 = self.backbone(x1)

        z0 = self.projection(f0)
        z1 = self.projection(f1)

        p0 = self.prediction(z0)
        p1 = self.prediction(z1)

        z0 = z0.detach()
        z1 = z1.detach()
        return (z0, p0), (z1, p1), f0

    def training_step(self, batch, batch_idx):
        (x0, x1), _, y = batch

        # print(x0.shape)
        # print(y.shape)

        (z0, p0), (z1, p1), embedding = self.forward(x0, x1)

        loss = 0.5 * (self.loss(z0, p1) + self.loss(z1, p0))

        # collapse based on https://docs.lightly.ai/tutorials/package/tutorial_simsiam_esa.html
        output = p0.detach()
        output = torch.nn.functional.normalize(output, dim=1)
        output_std = torch.std(output, 0)
        output_std = output_std.mean()

        # use moving averages to track the loss and standard deviation
        w = 0.9
        self.avg_loss = w * self.avg_loss + (1 - w) * loss.item()
        self.avg_output_std = w * self.avg_output_std + (
            1 - w) * output_std.item()

        self.logger.experiment.add_scalar('train_loss_ssl',  # type: ignore
                                          loss,
                                          global_step=self.current_epoch)
        self.logger.experiment.add_scalar('Avgloss',  # type: ignore
                                          self.avg_loss,
                                          global_step=self.current_epoch)
        self.logger.experiment.add_scalar('Avgstd',  # type: ignore
                                          self.avg_output_std,
                                          global_step=self.current_epoch)
        return {
            'loss': loss,
            'y_true': y.detach(),
            'embedding': embedding.detach()
        }

    def training_epoch_end(self, outputs):
        # the level of collapse is large if the standard deviation of the l2
        # normalized output is much smaller than 1 / sqrt(dim)
        self.collapse_level = max(
            0., 1 - math.sqrt(self.out_dim) * self.avg_output_std)
        # self.log('Collapse Level', round(self.collapse_level,2), logger=True)
        self.logger.experiment.add_scalar('Collapse Level',  # type: ignore
                                          round(self.collapse_level, 2),
                                          global_step=self.current_epoch)

        embedding_list = list()
        y_true_list = list()

        for item in outputs:
            embedding_list.append(item['embedding'])
            y_true_list.append(item['y_true'])

        # log every 10 epochs
        if not self.current_epoch % 10:
            self.logger.experiment.add_embedding(  # type: ignore
                torch.cat(embedding_list),
                metadata=torch.cat(y_true_list),
                global_step=self.current_epoch,
                tag='pretraining embedding')

    def validation_step(self, val_batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.epochs)
        return [optimizer], [scheduler]

