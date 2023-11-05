import lightly
import torch.nn as nn
from lightly.models.modules.heads import SimSiamPredictionHead


class SimSiam(nn.Module):
    """ Basic SimSiam without Pytorch Lightning."""

    def __init__(self,
                 backbone=nn.Module,
                 num_ftrs=64,
                 proj_hidden_dim=14,
                 pred_hidden_dim=14,
                 out_dim=14) -> None:
        super().__init__()
        self.backbone = backbone
        self.model_type: str = 'SimSiam nn.Module'
        self.projection = lightly.models.modules.heads.ProjectionHead([  # type: ignore
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

        return (z0, p0), (z1, p1)
