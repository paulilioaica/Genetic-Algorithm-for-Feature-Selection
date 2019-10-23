import time

import torch.nn as nn
import torch.optim as optim
import torch
import random

from dataloader import parse_dataset
from net import NNet
model = NNet(input_dim=13, num_of_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
X_train, Y_train, X_test, Y_test = parse_dataset()
losses = 0
for epoch in range(10):
    losses = 0
    for i, x in enumerate(X_train):
        optimizer.zero_grad()
        output = model(torch.tensor(x).unsqueeze(0))
        loss = criterion(output, torch.tensor(Y_train[i]).unsqueeze(0))
        losses += loss.item()
        loss.backward()
        optimizer.step()
    print(losses/len(X_train))




