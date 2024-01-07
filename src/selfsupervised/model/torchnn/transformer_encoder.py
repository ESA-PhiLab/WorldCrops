import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TransformerEncoder(nn.Module):
    """ Transformer Encoder with classification head. """
    def __init__(self,
                 input_dim=13,
                 num_classes=7,
                 d_model=64,
                 n_head=2,
                 d_ffn=128,
                 nlayers=2,
                 dropout=0.018,
                 activation="relu") -> None:
        super().__init__()
        """
        Args:
            input_dim: amount of input dimensions -> Sentinel2 has 13 bands
            num_classes: amount of target classes
            dropout: default = 0.018
            d_model: default = 64 #number of expected features
            n_head: default = 2 #number of heads in multiheadattention models
            d_ff: default = 128 #dim of feedforward network
            nlayers: default = 2 #number of encoder layers
            + : https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        Input:
            batch size(N) x T x D
        """

        self.model_type: str = 'Transformer encoder'
        self.inlinear = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU()
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model,
                                                    nhead=n_head,
                                                    dim_feedforward=d_ffn,
                                                    dropout=dropout,
                                                    activation=activation,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers, nn.LayerNorm(d_model))
        self.outlinear = nn.Linear(d_model, num_classes)

    def forward(self, x) -> Tensor:
        # N x T x D -> N x T x d_model / Batch First!
        x = self.inlinear(x)
        # torch.Size([N, T, d_model])
        x = self.relu(x)
        x = self.transformer_encoder(x)
        # torch.Size([N, T, d_model])
        x = x.max(1)[0]
        x = self.relu(x)
        x = self.outlinear(x)
        # torch.Size([N,num_classes ])
        x = F.log_softmax(x, dim=-1)
        # torch.Size([N, num_classes])
        return x

