from common import *
import numpy as np
import pandas as pd


POPULATION_SIZE = 10

GENERATIONS = 500
SOLUTION_FOUND = False

CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2

LOWER_BOUND = -5
UPPER_BOUND = 5


def sum_squares_compute_fitness(population, target):
    # Generate a 1D array of indexes from 1 to the individuals size
    indexes = np.arange(1, population.shape[1] + 1)
    # Calculate the result based on: sum(ix^2), for each individuals values in the population
    result = np.sum(indexes * ((np.abs(population[0:,:]) ** 2) * np.sign(population[0:,:])), axis=1)
    fitness = abs(result[0:,] - target) # Calculate the results absolute distance from desired value

    return fitness


def display_population(population):
    output_table = pd.DataFrame({'Individuals': population.tolist(),
                                 'Fitness':  sum_squares_compute_fitness(population, 0).tolist()})

    with pd.option_context('display.max_colwidth', -1, 'display.max_colwidth', 400):
        pd.set_option('colheader_justify', 'center')
        print(output_table)


def main():
    global POPULATION_SIZE
    global GENERATIONS
    global SOLUTION_FOUND

    global CROSSOVER_RATE
    global MUTATION_RATE

    global LOWER_BOUND
    global UPPER_BOUND

    population = generate_population(POPULATION_SIZE, 7, lower_bound, upper_bound)

    generation_counter = 0
    while (GENERATIONS > generation_counter):
        parents = selection_roulette(population, sum_squares_compute_fitness(population, 0), CROSSOVER_RATE)
        children = single_point_crossover(parents)
        children = uniform_mutation(children, lower_bound, upper_bound, MUTATION_RATE, 1)
        population = np.vstack((population, children))

        population = next_generation(population, sum_squares_compute_fitness(population, 0), POPULATION_SIZE)

        generation_counter += 1

        continue

    print('')
    print('   ##########################################################################################################################################')
    print('   ############################################################ FINAL GENERATION ############################################################')
    print('   ##########################################################################################################################################')
    display_population(population)


if __name__ == '__main__':
    main()
    
    
    
