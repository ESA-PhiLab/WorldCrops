import torch.nn as nn


class MLP(nn.Module):
    """ Multilayer Perceptron with 3 linear layers and Relu activation"""
    def __init__(self, input_dim, num_classes) -> None:
        super().__init__()
        self.input_dim: int = input_dim
        self.num_classes: int = num_classes

        self.layers = nn.Sequential(
            # convert input dim (width, height and channels)
            nn.Flatten(),
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes))

    def forward(self, x):
        return self.layers(x)

