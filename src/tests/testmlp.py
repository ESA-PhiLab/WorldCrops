import os

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from selfsupervised.model.lightning.mlp import MLP as MLP_lightning
from selfsupervised.model.torchnn.mlp import MLP

# Set CUDA_VISIBLE_DEVICES to an empty string
os.environ["CUDA_VISIBLE_DEVICES"] = ""

@pytest.fixture
def sample_data() -> tuple[torch.Tensor, torch.Tensor]:
    # Generate some random sample data for testing
    input_dim = 10
    num_classes = 5
    num_samples = 100
    x = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return x, y


def test_mlp(sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = sample_data

    # Set the device to CPU
    device = torch.device('cpu')

    # Create an instance of MLP for testing
    model = MLP_lightning(
        input_dim=x.shape[1], num_classes=max(y) + 1).to(device)

    # Dummy DataLoader for testing
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Dummy Lightning Trainer for testing
    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=1)

    # Fit the model
    trainer.fit(model, train_dataloaders=dataloader)

    # Test the model on a validation batch
    val_batch = next(iter(dataloader))

    # Make assertions based on your expectations
    assert model.training_step(sample_data) is not None
    assert model.validation_step(val_batch) is not None


def test_mlp_forward_pass(sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = sample_data

    # Create an instance of MLP for testing
    model = MLP(input_dim=x.shape[1], num_classes=max(y) + 1)

    # Ensure the forward pass works without errors
    y_pred = model.forward(x)

    # Check the output shape
    assert y_pred.shape[1] == model.num_classes


def test_mlp_layers_structure() -> None:
    # Create an instance of MLP for testing
    model = MLP(input_dim=10, num_classes=5)

    # Check the number and types of layers in the model
    expected_layers = [
        nn.Flatten,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
        nn.ReLU,
        nn.Linear
    ]

    actual_layers = [layer.__class__ for layer in model.layers]

    assert expected_layers == actual_layers


def test_mlp_training_step(sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = sample_data

    # Create an instance of MLP for testing
    model = MLP(input_dim=x.shape[1], num_classes=max(y) + 1)

    # Dummy loss function
    criterion = nn.CrossEntropyLoss()

    # Perform a forward and backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    # Ensure the training step completes without errors
    assert loss.item() is not None
