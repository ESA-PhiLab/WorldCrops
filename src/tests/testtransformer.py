import os

import pytest
import pytorch_lightning as pl
import torch

from selfsupervised.model.lightning.transformerencoder import \
    TransformerEncoder as TE2
from selfsupervised.model.torchnn.transformerencoder import \
    TransformerEncoder as TE1

os.environ["CUDA_VISIBLE_DEVICES"] = ""
batch_size = 2


@pytest.fixture
def sample_data():
    # Generate sample data for testing
    input_dim = 13
    sequence_length = 10
    batch_size = 2
    num_classes = 7
    x = torch.randn(batch_size, sequence_length, input_dim)
    y = torch.randint(0, num_classes, (batch_size,))
    return x, y


def test_transformer_encoder_forward_pass(sample_data):
    x, y = sample_data

    # Create an instance of TransformerEncoder for testing
    model = TE1()

    # Perform a forward pass
    outputs = model(x)

    # Check the shape of the output tensor

    assert outputs.shape == (batch_size, model.outlinear.out_features)

    # Check that the output tensor contains valid values (not NaN or Inf)
    assert torch.all(torch.isfinite(outputs))

    # Check that the output tensor contains valid probabilities
    assert torch.allclose(torch.sum(torch.exp(outputs),
                          dim=-1), torch.ones(x.shape[0]))

    # Check that the model returns a tensor of shape (batch_size, num_classes)
    assert outputs.shape == (batch_size, 7)


def test_transformer_encoder_lightning_forward_pass(sample_data):
    x, y = sample_data

    # Create an instance of TransformerEncoder for testing
    model = TE2()

    # Define a dummy DataLoader for testing
    dummy_dataloader = torch.utils.data.DataLoader(
        list(zip(x, y)), batch_size=batch_size)

    # Define a dummy Lightning Trainer for testing
    trainer = pl.Trainer(max_epochs=1, fast_dev_run=True)

    # Perform a forward pass using Lightning Trainer
    output = trainer.test(model, dataloaders=dummy_dataloader)

    assert output[0]['OA'] is not None
