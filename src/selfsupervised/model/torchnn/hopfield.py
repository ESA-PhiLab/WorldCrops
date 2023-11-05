
import torch
from torch import nn


class HopfieldLookup(nn.Module):
    """ Hopfield: https://ml-jku.github.io/hopfield-layers/"""
    def __init__(self, conf) -> None:
        super(HopfieldLookup, self).__init__()
        self.Wk = nn.Linear(conf['in_dim'],conf['hidden_dim'])
        self.Wv = nn.Linear(conf['hidden_dim'],conf['channels'])
        self.beta = conf['Hopfield_beta']
        self.soft = nn.Softmax(dim=1)
        self.classifier1 = nn.Sequential(
            nn.Linear(conf['channels'],int(conf['channels']/2)),
            nn.Linear(int(conf['channels']/2),conf['total_labels'])
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(conf['channels'],int(conf['channels']/2)),
            nn.Linear(int(conf['channels']/2),1)
        )
        self.softi = nn.Softmax(dim=1)

    def forward(self, R):
        R = R.float()
        k = self.Wk(R)
        y = self.soft(self.beta * k)
        out = self.Wv(y)
        pred = self.classifier1(out)
        pred = torch.einsum('abc->acb', pred)
        pred = self.classifier2(pred)[:,:,0]
        return self.softi(pred)


class HopfieldLookupClassifier(nn.Module):
    def __init__(self, conf) -> None:
        super(HopfieldLookupClassifier, self).__init__()
        self.Wq = nn.Linear(conf['in_dim'],conf['hidden_dim'])
        self.Wk = nn.Linear(conf['in_dim'],conf['hidden_dim'])
        self.Wv = nn.Linear(conf['channels'],conf['out_dim'])
        self.beta = conf['Hopfield_beta']
        self.soft = nn.Softmax(dim=1)
        self.classifier1 = nn.Sequential(
            nn.Linear(conf['out_dim'],int(conf['out_dim']/2)),
            nn.Linear(int(conf['out_dim']/2),conf['total_labels'])
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(conf['channels'],int(conf['channels']/2)),
            nn.Linear(int(conf['channels']/2),1)
        )
        self.softi = nn.Softmax(dim=1)

    def forward(self, R, Y=None):
        R = R.float()
        q = self.Wq(R)
        k = self.Wk(R)
        p = torch.einsum('zba,zca->zbc', q, k)
        y = self.soft(self.beta * p)
        out = self.Wv(y)
        pred = self.classifier1(out)
        pred = torch.einsum('abc->acb', pred)
        pred = self.classifier2(pred)[:,:,0]
        return self.softi(pred)


class HopfieldClassifier(nn.Module):
    def __init__(self, conf) -> None:
        super(HopfieldClassifier, self).__init__()
        self.Wq = nn.Linear(conf['in_dim'],conf['hidden_dim'])
        self.Wk = nn.Linear(conf['in_dim'],conf['hidden_dim'])
        self.beta = conf['Hopfield_beta']
        self.soft = nn.Softmax(dim=1)
        self.classifier1 = nn.Sequential(
            nn.Linear(conf['out_dim'],int(conf['out_dim']/2)),
            nn.Linear(int(conf['out_dim']/2),conf['total_labels'])
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(conf['out_dim'],int(conf['out_dim']/2)),
            nn.Linear(int(conf['out_dim']/2),1)
        )
        self.softi = nn.Softmax(dim=1)

    def forward(self, R, Y=None):
        R = R.float()
        q = self.Wq(R)
        k = self.Wk(R)
        p = torch.einsum('zba,zca->zbc', q, k)
        out = self.soft(self.beta * p)
        pred = self.classifier1(out)
        pred = torch.einsum('abc->acb', pred)
        pred = self.classifier2(pred)[:,:,0]
        return self.softi(pred)

