import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd


df = pd.read_csv('Desktop/covid19data/CountofCOVID-19casesbyepisodedateinOntario.csv')


def load_data(df):
    x, y = [], []

    for data, target in enumerate(df['Cases by episode date']):
        x.append(data)
        y.append(target)

    data = torch.from_numpy(np.array(
        x
    )).float()

    data = data.view(-1, 1)

    target = torch.from_numpy(np.array(
        y
    )).float()

    target = target.view(-1, 1)

    return DataLoader(TensorDataset(data, target), batch_size=32)
    

class QuadraticRegression(nn.Module):
    def __init__(
        self, in_features, out_features
    ):

        super(QuadraticRegression, self).__init__()

        self.a = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b = nn.Parameter(torch.Tensor(out_features, in_features))
        self.c = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    
    def forward(
        self, x
    ):

        return torch.sigmoid((self.a * (x ** 2)) + (self.b * x) + self.c)

    
    def reset_parameters(
        self
    ):
        nn.init.kaiming_uniform_(self.a, math.sqrt(5))
        nn.init.kaiming_uniform_(self.b, math.sqrt(5))
        
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.a)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.c, -bound, bound)


def optimize(epochs):
    data_loader = load_data(df)
    qr = QuadraticRegression(1, 1)
    optimizer = Adam(qr.parameters(), lr=1e-3)

    qr.train()

    for t in range(epochs):
        for x, y in data_loader:
            optimizer.zero_grad()
            pred = qr(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
        print("Epoch: {}".format(t))

    return qr


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    qr = optimize(1000)
    qr.eval()

    x, y = [], []

    for data, target in enumerate(df['Cases by episode date']):
        x.append(data), y.append(target)

    x_qr = [torch.tensor(data).view(1, 1).float() for data in x]
    y_qr = [qr(data).view(1).detach().numpy() for data in x_qr]

    plt.plot(x, y, 'r')
    plt.plot(x, y_qr)
    plt.show()
