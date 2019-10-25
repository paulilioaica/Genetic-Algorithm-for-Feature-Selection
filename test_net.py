import time

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import random

best_input = [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0]
best_input = [1 for i in range(13)]
from dataloader import parse_dataset
from net import NNet

model = NNet(input_dim=best_input.count(1), num_of_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
X_train, Y_train, X_test, Y_test = parse_dataset()
losses = 0
for epoch in range(100):
    losses = 0
    for i, x in enumerate(X_train):
        x_local = [value for idx, value in enumerate(x) if best_input[idx] == 1]
        optimizer.zero_grad()
        output = model(torch.tensor(x_local).unsqueeze(0).float())
        loss = criterion(output, torch.tensor(Y_train[i]).unsqueeze(0))
        losses += loss.item()
        loss.backward()
        optimizer.step()
    outputs = []
    for i, x in enumerate(X_test):
        model.eval()
        x_local = [value for idx, value in enumerate(x) if best_input[idx] == 1]
        output = model(torch.tensor(x_local).unsqueeze(0).float())
        loss = criterion(output, torch.tensor(Y_test[i]).unsqueeze(0))
        outputs.append(torch.argmax(output).item())
    accuracy = len(np.where(np.array(Y_test) == np.array(outputs))[0]) / len(Y_test)
    print(f"Accuracy is {accuracy * 100}%")
    print(losses / len(X_train))
