import time

import torch.nn as nn
import torch.optim as optim
import torch
import random

import numpy as np

from net import NNet
from dataloader import parse_dataset

POPULATION_SIZE = 10
EPOCHS = 2
CHROMOSOME_SIZE = 13
NUM_OFFSPRING = 2

population = [[random.choice([1, 0]) for i in range(13)] for i in range(POPULATION_SIZE)]
criterion = nn.CrossEntropyLoss()

X_train, Y_train, X_test, Y_test = parse_dataset()


def create_nn_array(sizes):
    neural_nets = [NNet(sizes[i], num_of_classes=3) for i in range(len(sizes))]
    optimizers = [optim.Adam(net.parameters()) for net in neural_nets]
    return neural_nets, optimizers

def check_population(sizes, population):
    for i, size in enumerate(sizes):
        if size == 0 :
            population[i] = [1 for i in range(POPULATION_SIZE)]
    return population
def get_sizes(population):
    return [x.count(1) for x in population]


def train(neural_nets, optimizers):
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
                optimizers[j].zero_grad()
                x_local = [value for idx, value in enumerate(x_local) if population[j][idx] == 1]
                output = net(torch.tensor(x_local).unsqueeze(0))
                loss = criterion(output, torch.tensor(Y_train[i]).unsqueeze(0))
                losses[j].append(loss.item())
                loss.backward()
                optimizers[j].step()
    losses = [sum(x) / len(x) for x in losses]
    print(min(losses))
    return losses


def selection(losses):
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


def get_probab(losses):
    losses = [x / sum(losses) for x in losses]
    return losses


def run_genetic_algorithm():
    while True:
        global population
        sizes = get_sizes(population)
        nets, optimizers = create_nn_array(sizes)
        losses = train(nets, optimizers)
        losses = get_probab(losses)
        first_candidate, second_candidate = selection(losses)
        first_offspring, second_offspring = cross_over(population[first_candidate], population[second_candidate])
        first_offspring = mutation(first_offspring)
        population = [first_offspring]
        population += [mutation(first_offspring) for i in range(POPULATION_SIZE - 1)]
        population = check_population(sizes, population)

run_genetic_algorithm()