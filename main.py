import random

import numpy as np

from dataloader import parse_dataset
import matplotlib.pyplot as plt
from genetic import train, selection, cross_over, mutation, check_population
from utils import get_sizes, create_nn_array, get_probab

random.seed(99)
POPULATION_SIZE = 10
EPOCHS = 5
CHROMOSOME_SIZE = 13
NUM_OFFSPRING = 2
population = [[random.choice([1, 0]) for i in range(13)] for i in range(POPULATION_SIZE)]
X_train, Y_train, X_test, Y_test = parse_dataset()
accuracies = []
chromosomes = []

features = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']


def run_genetic_algorithm():
    generation = 0
    global population
    for i in range(200):
        print("Training generation {}".format(generation))
        sizes = get_sizes(population) # get the sizes of the new populations for the networks inputs
        nets = create_nn_array(sizes)  # we create a list of neural networks with predefined size
        losses, BEST_CHROMOSOME, max_accuracy = train(nets, X_train, Y_train, X_test, Y_test, population)# we train EPOCHS number for each network
        losses = get_probab(losses) # we select the one with the smallest loss
        first_candidate, second_candidate = selection(losses)
        first_offspring, second_offspring = cross_over(population[first_candidate], population[second_candidate]) # cross over between top 2 candidates
        first_offspring = mutation(first_offspring) # we mutate the result
        population = [first_offspring]
        population += [mutation(first_offspring) for i in range(POPULATION_SIZE - 1)] # create a population of individuals based on the best candidates
        population = check_population(sizes, population) # we make sure no inidividual is only 0's
        generation += 1
        accuracies.append(max_accuracy)
        if BEST_CHROMOSOME:
            chromosomes.append(BEST_CHROMOSOME)


run_genetic_algorithm()
