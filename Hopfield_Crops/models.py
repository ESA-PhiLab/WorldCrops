#%%
import torch
from torch import nn
import numpy as np
import pandas as pd
# DATA 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# class Hopfield_v1(nn.Module):
#     def __init__(self, conf):
#         super(Hopfield_v1, self).__init__()
#         self.Wq = nn.Linear(conf['in_dim'],conf['hidden_dim'])
#         self.Wk = nn.Linear(conf['in_dim'],conf['hidden_dim'])
#         self.Wv = nn.Linear(conf['in_dim'],conf['channels'])
#         self.beta = conf['Hopfield_beta']
#         self.soft = nn.Softmax(dim=1)

#     def forward(self, R, Y=None):
#         R = R.float()
#         q = self.Wq(R)
#         k = self.Wk(Y)
#         p = torch.einsum('za,zca->zc', q, k)
#         y = self.soft(self.beta * p)
#         v = self.Wv(Y)
#         out = torch.einsum('zb,zbc->zc', y, v)
#         return out

#     def spectrum(self, R, Y):
#         return self.forward(R, Y)


class Hopfield_v1_Lookup(nn.Module):
    def __init__(self, conf):
        super(Hopfield_v1_Lookup, self).__init__()
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

    def forward(self, R, Y):
        R = R.float()
        k = self.Wk(R)
        y = self.soft(self.beta * k)
        out = self.Wv(y)
        pred = self.classifier1(out)
        pred = torch.einsum('abc->acb', pred)
        pred = self.classifier2(pred)[:,:,0]
        return self.softi(pred)

    def spectrum(self, R):
        return self.forward(R)


class Hopfield_v1_LookupClassifier(nn.Module):
    def __init__(self, conf):
        super(Hopfield_v1_LookupClassifier, self).__init__()
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

    def spectrum(self, R, Y):
        return self.forward(R, Y)



class Hopfield_v1_Classifier(nn.Module):
    def __init__(self, conf):
        super(Hopfield_v1_Classifier, self).__init__()
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

    def spectrum(self, R, Y):
        return self.forward(R, Y)



# class Hopfield_v2_Classifier(nn.Module):
#     def __init__(self, conf):
#         super(Hopfield_v2_Classifier, self).__init__()
#         self.E = nn.Linear(conf['in_dim'], conf['emb_dim'])
#         self.Wq = nn.Linear(conf['emb_dim'],conf['hidden_dim'])
#         self.Wk = nn.Linear(conf['emb_dim'],conf['hidden_dim'])
#         self.beta = conf['Hopfield_beta']
#         self.soft = nn.Softmax(dim=1)
#         self.classifier1 = nn.Sequential(
#             nn.Linear(conf['emb_dim'],int(conf['emb_dim']/2)),
#             nn.Linear(int(conf['emb_dim']/2),conf['total_labels'])
#         )
#         self.classifier2 = nn.Sequential(
#             nn.Linear(conf['emb_dim'],int(conf['emb_dim']/2)),
#             nn.Linear(int(conf['emb_dim']/2),1)
#         )
#         self.softi = nn.Softmax(dim=1)

#     def forward(self, R, Y=None):
#         R = R.float()
#         print(R.shape)
#         Re = self.E(R)
#         print(Re.shape)
#         q = self.Wq(Re)
#         k = self.Wk(Re)
#         p = torch.einsum('zba,zca->zbc', q, k)
#         out = self.soft(self.beta * p)
#         print(out.shape)
#         exit()
#         pred = self.classifier1(out)
#         pred = torch.einsum('abc->acb', pred)
#         pred = self.classifier2(pred)[:,:,0]
#         return self.softi(pred)

#     def spectrum(self, R, Y):
#         return self.forward(R, Y)


# ### LIBRARY LAYER MULTI HEAD
# class Hopfield_v4_layer(nn.Module):
#     def __init__(self, conf):
#         super(Hopfield_v4_layer, self).__init__()
#         self.conf = conf
#         self.heads = conf['hop_heads']
#         self.Wq = nn.Linear(conf['in_dim'], self.heads*conf['hidden_dim'])
#         self.Wk = nn.Linear(conf['in_dim'], self.heads*conf['hidden_dim'])
#         self.Wv = nn.Linear(conf['in_dim'], self.heads*conf['out_dim'])
#         self.beta = conf['Hopfield_beta']
#         self.soft = nn.Softmax(dim=3)
#         self.Condens = nn.Sequential(
#             nn.Linear(self.heads*conf['in_dim'], int(np.ceil(self.heads/2))*conf['in_dim']),
#             nn.Linear(int(np.ceil(self.heads/2))*conf['in_dim'], conf['in_dim'])
#         )


#     def forward(self, data):
#         emb, Y = data[0], data[1]
#         q = self.Wq(emb)
#         k = self.Wk(Y)
#         v = self.Wv(Y)
#         #                       BATCH     EMBEDDING      HEADS       HIDDEN DIMENSION
#         q = torch.reshape(q, (q.shape[0], q.shape[1], self.heads, self.conf['hidden_dim']))
#         #                       BATCH        HEADS       LENGTH      HIDDEN DIMENSION
#         k = torch.reshape(k, (k.shape[0], self.heads, k.shape[1], self.conf['hidden_dim']))
#         #                       BATCH        HEADS       LENGTH       OUT DIMENSION
#         v = torch.reshape(v, (v.shape[0], self.heads, v.shape[1], self.conf['out_dim']))
#         p = torch.einsum('beza,bzca->bzec', q, k)
#         y = self.soft(self.beta * p)
#         out = torch.einsum('bzea,bzac->bzce', y, v)
#         out = torch.reshape(out, (out.shape[0], self.heads*self.conf['in_dim'], self.conf['emb_dim']))
#         out = torch.einsum('abc->acb', out)
#         out = self.Condens(out)
#         return [out, Y]

# class Hopfield_v4(nn.Module):
#     def __init__(self, conf):
#         super(Hopfield_v4, self).__init__()
#         self.conf = conf
#         self.heads = conf['hop_heads']
#         self.layers = conf['hop_layers']
#         self.E = nn.Linear(1, conf['emb_dim'])
#         self.Hopfield_Layer = nn.Sequential()
#         for l in range(self.layers):
#             self.Hopfield_Layer.add_module(str(l), Hopfield_v4_layer(conf))
#         self.iE = nn.Linear(conf['emb_dim'], 1)

#     def forward(self, R, Y):
#         R_e = torch.einsum('ab->ba', R)
#         emb = self.E(R_e[:,:,None])
#         emb = torch.einsum('abc->bca', emb)
#         out = self.Hopfield_Layer([emb, Y])[0]
#         out = torch.einsum('abc->acb', out)
#         out = self.iE(out)#[:,:,0]
#         out = torch.reshape(out, (out.shape[0], -1))
#         return out

#     def spectrum(self, R, Y):
#         return self.forward(R, Y)







########### DATA LOADER 
class Data(Dataset):
    def __init__(self, year):
        data = np.load('Bavaria_13.npy', allow_pickle=True).item()
        self.data = data['data']
        # print(self.data.shape)
        self.target = data['target']
        tot_samples = int(self.data.shape[1] * self.data.shape[2])
        tot_time_steps = self.data.shape[4]
        tot_channels = self.data.shape[3]
        print(' ')
        print('--------------------------------')
        print(' >>> DATA SUMMARY ')
        print('Total Samples:', tot_samples)
        print('Time Steps:', tot_time_steps)
        print('Channels:', tot_channels)
        print('--------------------------------')
        print(' ')
        self.data = np.reshape(self.data[year], (len(year)*tot_samples, tot_channels, tot_time_steps))
        self.target = np.reshape(self.target[year], (len(year)*tot_samples, 6))
        # self.data = np.einsum('abc->acb', self.data)
        # print(self.data.shape)
        # exit()


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.target[idx]

def get_dataloader(year, batch_size, shuffle=False, RAW_PATH=None):
    set = Data(year)
    return DataLoader(set, batch_size=batch_size, shuffle=shuffle)


# #%%
# data = get_dataloader(year=[1,2], batch_size=1000)

#%%
# print(data)
# for sample in data:
#     print(sample[0].shape, sample[1].shape)
#     # break


