import lightly
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lightly.models.modules import NNMemoryBankModule
from lightly.models.modules.heads import ProjectionHead
from lightly.models.utils import batch_shuffle
from lightly.models.utils import batch_unshuffle
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.utils import BenchmarkModule

from lightly.models.modules.heads import SimSiamPredictionHead
from lightly.models.modules.heads import SimSiamProjectionHead

class SimSiam(pl.LightningModule):
    def __init__(
        self, backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim, lr=6e-2,
        momentum=0.9, weight_decay=5e-4, epochs=5
    ):
        super().__init__()
        """
        Args:
        Input:

        """
        self.model_type = 'SimSiam'
        self.weight_decay =weight_decay
        self.momentum = momentum
        self.lr = lr
        self.epochs = epochs

        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(
            num_ftrs, proj_hidden_dim, out_dim
        )
        self.prediction_head = SimSiamPredictionHead(
            out_dim, pred_hidden_dim, out_dim
        )
        self.ce = lightly.loss.negative_cosine_similarity.NegativeCosineSimilarity()

    def forward(self, x):
        # encoder
        f = self.backbone(x).flatten(start_dim=1)
        # embeddings
        z = self.projection_head(f)
        # predictions
        p = self.prediction_head(z)
        # stop gradient
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0 = torch.unsqueeze(x0, 1)
        x1 = torch.unsqueeze(x1, 1)
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.ce(z0, p1) + self.ce(z1, p0))
        self.log('train_loss_ssl', loss)
        return {'loss':loss,'p0':p0}

    def training_step_end(self, outputs):
        #outputs = torch.cat(outputs, dim=1)
        #softmax = softmax(outputs, dim=1)
        #out = softmax.mean()
        output = outputs['p0'].detach()
        output = torch.nn.functional.normalize(output, dim=1)
        output_std = torch.std(output, 0)
        output_std = output_std.mean()

        # use moving averages to track the loss and standard deviation
        w = 0.9
        avg_loss = w * avg_loss + (1 - w) * loss.item()
        avg_output_std = w * avg_output_std + (1 - w) * output_std.item()
        return avg_loss, avg_output_std
    
    def training_epoch_end(self, outputs ):
        avg_loss, avg_output_std = self.training_step_end(self, outputs)
        collapse_level = max(0., 1 - math.sqrt(out_dim) * avg_output_std)
        self.log('loss', round(avg_loss,2), prog_bar=True)
        self.log('Collapse Level', round(collapse_evel,2), prog_bar=True)


    def validation_step(self, val_batch, batch_idx):
        (x0, x1), _, _ = batch
        x0 = torch.unsqueeze(x0, 1)
        x1 = torch.unsqueeze(x1, 1)
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.ce(z0, p1) + self.ce(z1, p0))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,
                                momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        return [optimizer], [scheduler]

#add this
#https://githubmemory.com/repo/PyTorchLightning/pytorch-lightning/issues/8302

class SimSiam_1D(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
