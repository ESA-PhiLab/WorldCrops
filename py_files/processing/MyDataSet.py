import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, input, seq_len):
        self.input = input
        self.seq_len = seq_len
    def __getitem__(self, item):
        return input[item:item + self.seq_len], input[item + self.seq_len]
    def __len__(self):
        return len(self.input) - self.seq_len