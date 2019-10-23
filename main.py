import random
from dataloader import parse_dataset
from genetic import train, selection, cross_over, mutation, check_population
from utils import get_sizes, create_nn_array, get_probab
POPULATION_SIZE = 10
EPOCHS = 2
CHROMOSOME_SIZE = 13
NUM_OFFSPRING = 2

population = [[random.choice([1, 0]) for i in range(13)] for i in range(POPULATION_SIZE)]

X_train, Y_train, X_test, Y_test = parse_dataset()

def run_genetic_algorithm():
    global population
    while True:
        sizes = get_sizes(population)
        nets, optimizers = create_nn_array(sizes)
        losses = train(nets, optimizers, X_train, Y_train, X_test, Y_test, population)
        losses = get_probab(losses)
        first_candidate, second_candidate = selection(losses)
        first_offspring, second_offspring = cross_over(population[first_candidate], population[second_candidate])
        first_offspring = mutation(first_offspring)
        population = [first_offspring]
        population += [mutation(first_offspring) for i in range(POPULATION_SIZE - 1)]
        population = check_population(sizes, population)

run_genetic_algorithm()