import torch.nn as nn
# ensure lightly is installed
from lightly.models.modules.heads import ProjectionHead, SimSiamPredictionHead


class SimSiam(nn.Module):
    """
    Basic implementation of SimSiam architecture without Pytorch Lightning.

    SimSiam is a framework for unsupervised learning of visual representations. It
    learns representations by maximizing the similarity between two augmented views
    of the same image / time series, without using negative pairs.

    Attributes:
        backbone (nn.Module): The backbone acts as the feature extractor.
        model_type (str): Identifier for the model type.
        projection (lightly.models.modules.heads.ProjectionHead): The projection
                     head that maps features extracted by the backbone to a lower-dimensional
                     space.
        prediction (SimSiamPredictionHead): The prediction head of the SimSiam model
                    which predicts one view's representation from the other view's
                    projected representation.

    Args:
        backbone (nn.Module): The backbone model for feature extraction. Defaults to
                              a general nn.Module.
        num_ftrs (int): Number of features in the input tensor expected by the
                        projection head.
        proj_hidden_dim (int): Hidden layer dimension in the projection head.
        pred_hidden_dim (int): Hidden layer dimension in the prediction head.
        out_dim (int): Output dimension for both the projection and prediction heads.

    Forward pass inputs:
        x0 (Tensor): The first input tensor (one view).
        x1 (Tensor): The second input tensor (another view).

    Returns:
        Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]: A tuple containing two tuples,
        each with the projected and predicted representations of the input views.
        Specifically, ((z0, p0), (z1, p1)), where z0 and z1 are the projected features of x0
        and x1,  and p0 and p1 are the predicted features based on z0 and z1.
    """

    def __init__(self, backbone=nn.Module, num_ftrs=64, proj_hidden_dim=14,
                 pred_hidden_dim=14, out_dim=14) -> None:
        super().__init__()
        self.backbone = backbone
        self.model_type: str = 'SimSiam nn.Module'
        self.projection = ProjectionHead([  # type: ignore
            (num_ftrs, proj_hidden_dim, nn.BatchNorm1d(proj_hidden_dim), nn.ReLU()),
            (proj_hidden_dim, out_dim, nn.BatchNorm1d(out_dim), None)
        ])
        self.prediction = SimSiamPredictionHead(
            out_dim, pred_hidden_dim, out_dim)

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
