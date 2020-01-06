import random

import numpy as np

random.seed(99)

POPULATION_SIZE = 10
EPOCHS = 30
CHROMOSOME_SIZE = 13
NUM_OFFSPRING = 2
max_accuracy = 0
epoch_accuracy = [0 for i in range(EPOCHS)]
BEST_CHROMOSOME = None


def check_population(sizes, population):
    for i, size in enumerate(sizes):
        if size == 0:
            population[i] = [1 for i in range(POPULATION_SIZE)]
    return population


def train(neural_nets, X_train, Y_train, X_test, Y_test, population):
    # train EPOCHS for each network and test it on test data
    global max_accuracy
    global BEST_CHROMOSOME
    losses = [[] for i in range(POPULATION_SIZE)]
    accuracy = [[] for i in range(POPULATION_SIZE)]
    for epoch in range(EPOCHS):

        for j, net in enumerate(neural_nets):
            x_local = np.copy(X_train)
            position = []
            for idx in range(X_train.shape[1]):
                if population[j][idx] == 0:
                    position.append(idx)

            x_local = np.delete(x_local, position, 1)
            print(x_local.shape)
            print(net.weights1.shape)
            net.train(x_local, Y_train)

        for j, net in enumerate(neural_nets):
            x_local = np.copy(X_test)
            position = []
            for idx in range(X_train.shape[1]):
                if population[j][idx] == 0:
                    position.append(idx)
            x_local = np.delete(x_local, position, 1)
            output = net.predict(x_local)
            loss = np.square(np.argmax(output, axis=1) - Y_test).mean()
            acc = len(np.where(np.argmax(output, axis=1) == Y_test)[0]) / len(Y_test)
            accuracy[j].append(acc)
            losses[j].append(loss)


    avg_accuracy = [sum(x) / len(x) for x in accuracy]
    losses = [sum(x) / len(x) for x in losses]
    if max_accuracy < max(avg_accuracy):
        max_idx = np.argmax(avg_accuracy)
        max_accuracy = max(avg_accuracy)
        BEST_CHROMOSOME = population[max_idx]
        print(BEST_CHROMOSOME)
        epoch_accuracy[epoch] = avg_accuracy
        return losses, [x for x in BEST_CHROMOSOME], max_accuracy

    epoch_accuracy[epoch] = avg_accuracy
    return losses, None, max_accuracy


def selection(losses):
    ##### we select top 2 individuals to cross breed ############
    first_candidate, second_candidate = (-np.array(losses)).argsort()[:2]
    return first_candidate, second_candidate


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
