import pytest
import torch
import torch.nn as nn

from selfsupervised.model.lightning.simsiam import SimSiam as LigthningSimSiam
from selfsupervised.model.lightning.transformerencoder import \
    TransformerEncoder as TE2
from selfsupervised.model.torchnn.simsiam import SimSiam as ModuleSimSiam
from selfsupervised.model.torchnn.transformerencoder import \
    TransformerEncoder as TE1
