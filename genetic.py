import random
import numpy as np
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
POPULATION_SIZE = 10
EPOCHS = 5
CHROMOSOME_SIZE = 13
NUM_OFFSPRING = 2
max_accuracy = 0
criterion = nn.CrossEntropyLoss()
epoch_accuracy = [0 for i in range(EPOCHS)]
BEST_CHROMOSOME = None

def check_population(sizes, population):
    for i, size in enumerate(sizes):
        if size == 0:
            population[i] = [1 for i in range(POPULATION_SIZE)]
    return population


def train(neural_nets, optimizers, X_train, Y_train, X_test, Y_test, population):
    global BEST_CHROMOSOME
    global max_accuracy
    losses = [[] for i in range(POPULATION_SIZE)]
    ########### train ##################
    for epoch in range(EPOCHS):
        output_val = [[] for i in range(POPULATION_SIZE)]
        for i, x in enumerate(X_train):
            for j, net in enumerate(neural_nets):
                x_local = [item for item in x]
                optimizers[j].zero_grad()
                x_local = [value for idx, value in enumerate(x_local) if population[j][idx] == 1]
                output = net(torch.tensor(x_local).unsqueeze(0).to(device))
                loss = criterion(output, torch.tensor(Y_train[i]).unsqueeze(0).to(device))
                loss.backward()
                optimizers[j].step()
        for i, x in enumerate(X_test):
            for j, net in enumerate(neural_nets):
                x_local = [item for item in x]
                x_local = [value for idx, value in enumerate(x_local) if population[j][idx] == 1]
                net.eval()
                output = net(torch.tensor(x_local).unsqueeze(0).to(device))
                loss = criterion(output, torch.tensor(Y_test[i]).unsqueeze(0).to(device))
                output_val[j].append(torch.argmax(output).item())
                losses[j].append(loss.item())

        avg_accuracy =[len(np.where(np.array(x) == np.array(Y_test))[0])/len(Y_test) for x in output_val]
        if max_accuracy < max(avg_accuracy):
            max_accuracy = max(avg_accuracy)
            BEST_CHROMOSOME = population[np.argmax(np.array(max_accuracy))]
        print(BEST_CHROMOSOME)
        print(f"Average accuracy for this epoch is {avg_accuracy}%")
        print(f"Maximum is {max_accuracy}")
        print('\n')
        epoch_accuracy[epoch] = avg_accuracy

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
