from typing import Any

import torch.nn as nn


class ThreeLayerHead(nn.Module):
    """ MLP Head """
    def __init__(self, input_dim, num_classes) -> None:
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
        self.output_dim = num_classes

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, int(self.input_dim / 2)), nn.ReLU(),
            nn.Linear(int(self.input_dim / 2), int(self.input_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(self.input_dim / 2), self.output_dim))

    def forward(self, x) -> Any:
        return self.layers(x)
