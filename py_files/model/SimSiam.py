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

class SimSiam(nn.Module):
    def __init__(self, backbone = nn.Module, num_ftrs=64, proj_hidden_dim=14, pred_hidden_dim=14, out_dim=14):
        super().__init__()
        self.backbone = backbone
        self.model_type = 'SimSiam_nn.Module'
        self.projection = lightly.models.modules.heads.ProjectionHead([
            (num_ftrs, proj_hidden_dim, nn.BatchNorm1d(proj_hidden_dim), nn.ReLU()),
            (proj_hidden_dim, out_dim, nn.BatchNorm1d(out_dim), None)
        ])
        self.prediction = SimSiamPredictionHead(out_dim,pred_hidden_dim,out_dim)


    def forward(self, x0, x1):
        f0 = self.backbone(x0)
        f1 = self.backbone(x1)

        z0 = self.projection(f0)
        z1 = self.projection(f1)

        p0 = self.prediction(z0)
        p1 = self.prediction(z1)

        z0 = z0.detach()
        z1 = z1.detach()
        
        return (z0, p0),(z1, p1)



class SimSiam_LM(pl.LightningModule):
    def __init__(self, backbone = nn.Module, num_ftrs=64, proj_hidden_dim=14, 
    pred_hidden_dim=14, out_dim=14, lr=0.02, weight_decay=5e-4,momentum=0.9,epochs = 10):
        super().__init__()
        
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.backbone = backbone
        self.model_type = 'SimSiam_LM'
        self.projection = lightly.models.modules.heads.ProjectionHead([
            (num_ftrs, proj_hidden_dim, nn.BatchNorm1d(proj_hidden_dim), nn.ReLU()),
            (proj_hidden_dim, out_dim, nn.BatchNorm1d(out_dim), None)
        ])
        self.prediction = SimSiamPredictionHead(out_dim,pred_hidden_dim,out_dim)

        self.ce = lightly.loss.NegativeCosineSimilarity()

        self.avg_loss = 0.
        self.avg_output_std = 0.
        

    def forward(self, x0, x1):
        f0 = self.backbone(x0)
        f1 = self.backbone(x1)

        z0 = self.projection(f0)
        z1 = self.projection(f1)

        p0 = self.prediction(z0)
        p1 = self.prediction(z1)

        z0 = z0.detach()
        z1 = z1.detach()
        return (z0, p0),(z1, p1)


    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        (z0, p0),(z1, p1) = self.forward(x0,x1)

        loss = 0.5 * (self.ce(z0, p1) + self.ce(z1, p0))
        self.log('train_loss_ssl', loss)
        #output = p0.detach()
        return {'loss':loss}

    '''def training_step_end(self, batch_parts):
        #print(batch_parts['p0'], type(batch_parts['p0']))
        output = batch_parts['p0']
        loss = batch_parts['loss']
        output = torch.nn.functional.normalize(output, dim=1)
        output_std = torch.std(output, 0)
        output_std = output_std.mean()
        
        # use moving averages to track the loss and standard deviation
        w = 0.9
        self.avg_loss = w * self.avg_loss + (1 - w) * loss
        self.avg_output_std = w * self.avg_output_std + (1 - w) * output_std.item()
        return {'loss':self.avg_loss,'avg_output_std': self.avg_output_std}
    
    def training_epoch_end(self, training_step_outputs):

        collapse_level = max(0., 1 - math.sqrt(14) * self.avg_output_std)
        self.log('loss', round(self.avg_loss,2), prog_bar=True)
        self.log('Collapse Level', round(collapse_evel,2), prog_bar=True)'''


    def validation_step(self, val_batch, batch_idx):
        (x0, x1), _, _ = val_batch
 
        (z0, p0),(z1, p1) = self.forward(x0,x1)
        loss = 0.5 * (self.ce(z0, p1) + self.ce(z1, p0))
        self.log('val_loss', loss)
        return {"loss":loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,
                                momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        return [optimizer], [scheduler]

#add this
#https://githubmemory.com/repo/PyTorchLightning/pytorch-lightning/issues/8302

