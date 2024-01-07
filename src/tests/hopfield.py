import pytest
import torch

from selfsupervised.model.torchnn.hopfield import (HopfieldClassifier,
                                                   HopfieldLookup,
                                                   HopfieldLookupClassifier)


@pytest.fixture
def sample_input():
    # samples x channels x input dim
    return torch.randn((1, 16,64))


def test_hopfield_lookup_dimensions(sample_input):
    conf = {
        'in_dim': 64,
        'hidden_dim': 32,
        'channels': 16,
        'Hopfield_beta': 0.5,
        'total_labels': 10
    }
    model = HopfieldLookup(conf)
    output = model(sample_input)

    #print(output)
    assert output.squeeze(0).shape == torch.Size([10])


def test_hopfield_lookup_classifier_dimensions(sample_input):
    conf = {
        'in_dim': 64,
        'hidden_dim': 32,
        'channels': 16,
        'out_dim': 8,
        'Hopfield_beta': 0.5,
        'total_labels': 10
    }
    model = HopfieldLookupClassifier(conf)
    output = model(sample_input)

    assert output.squeeze(0).shape == torch.Size([10])
