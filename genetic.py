import random

import numpy as np
import torch
from torch import nn

POPULATION_SIZE = 10
EPOCHS = 2
CHROMOSOME_SIZE = 13
NUM_OFFSPRING = 2
criterion = nn.CrossEntropyLoss()

def check_population(sizes, population):
    for i, size in enumerate(sizes):
        if size == 0:
            population[i] = [1 for i in range(POPULATION_SIZE)]
    return population


def train(neural_nets, optimizers, X_train, Y_train, X_test, Y_test, population):
    losses = [[] for i in range(POPULATION_SIZE)]
    ########### train ##################
    for epoch in range(EPOCHS):
        for i, x in enumerate(X_train):
            for j, net in enumerate(neural_nets):
                x_local = [item for item in x]
                optimizers[j].zero_grad()
                x_local = [value for idx, value in enumerate(x_local) if population[j][idx] == 1]
                output = net(torch.tensor(x_local).unsqueeze(0))
                loss = criterion(output, torch.tensor(Y_train[i]).unsqueeze(0))
                loss.backward()
                optimizers[j].step()
        for i, x in enumerate(X_test):
            for j, net in enumerate(neural_nets):
                x_local = [item for item in x]
                x_local = [value for idx, value in enumerate(x_local) if population[j][idx] == 1]
                net.eval()
                output = net(torch.tensor(x_local).unsqueeze(0))
                loss = criterion(output, torch.tensor(Y_test[i]).unsqueeze(0))
                losses[j].append(loss.item())
    losses = [sum(x) / len(x) for x in losses]
    print(min(losses))
    return losses


def selection(losses: object) -> object:
    ##### we select top 2 individuals to cross breed ############
    first_candidate, second_candidate = random.choices(losses, losses, k=2)
    return losses.index(first_candidate), losses.index(second_candidate)


def cross_over(first_candidate, second_candidate):
    random_point = np.random.randint(0, CHROMOSOME_SIZE)
    for i in range(CHROMOSOME_SIZE):
        if i > random_point:
            first_candidate[i], second_candidate[i] = second_candidate[i], first_candidate[i]
    return first_candidate, second_candidate


def mutation(offspring):
    for i in range(len(offspring)):
        if np.random.rand() > 0.9:
            offspring[i] = int(not offspring[i])

    return offspring
