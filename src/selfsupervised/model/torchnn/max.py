import torch
import torch.nn as nn


class Max(nn.Module):
    """
    Custom PyTorch module that computes the maximum values along a specified dimension of an input tensor.

    Args:
        dim (int, optional): The dimension along which to compute the maximum values. 
                             Defaults to 1.

    Example:
        max_layer = Max(dim=2)
        output = max_layer(input_tensor)
    """

    def __init__(self, dim=1):
        super(Max, self).__init__()
        self.dim = dim

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.dim >= x.dim():
            raise ValueError(
                f"Dimension {self.dim} out of range for input with {x.dim()} dimensions.")

        return x.max(self.dim)[0]
