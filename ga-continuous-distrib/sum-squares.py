from common import *
import numpy as np


POPULATION_SIZE = 10000 # The maximum size of the population for each generation
INDIVIDUAL_SIZE = 7 # The number genes in each individual in the population

LOWER_BOUND = -5 # The lower limit that a gene value can be
UPPER_BOUND = 5 # The upper limit that a gene value can be

TARGET = 0 # The target value for fitness to be calculated, set to 0 for minimisation solution

GENERATIONS = 100 # The number of generations to run (if using as termination condition)
SOLUTION_FOUND = False # Whether an exact solution has been found (if using as termination condition)

CROSSOVER_RATE = 0.8 # The proportion of the population that will crossover to produce offspring each generation
MUTATION_RATE = 0.2 # The chance each offspring has of a gene (or multiple genes) being mutated each generation
MUTATIONS = 1 # The number of genes that are mutated if an offspring is selected for mutation (can be randomised with limits)


def sum_squares_compute_fitness(population, target=0):
    # Generate a 1D array of indexes from 1 to the individuals size
    indexes = np.arange(1, population.shape[1] + 1)
    # Calculate the result based on: sum(ix^2), for each individuals values in the population
    result = np.sum(indexes * ((np.abs(population[0:,:]) ** 2) * np.sign(population[0:,:])), axis=1)
    fitness = abs(result[0:,] - target) # Calculate the results absolute distance from desired value

    return fitness


def main():
    global POPULATION_SIZE
    global INDIVIDUAL_SIZE

    global LOWER_BOUND
    global UPPER_BOUND

    global TARGET

    global GENERATIONS
    global SOLUTION_FOUND

    global CROSSOVER_RATE
    global MUTATION_RATE
    global MUTATIONS

    # Generate initial population given parameters
    population = generate_population(POPULATION_SIZE, INDIVIDUAL_SIZE, LOWER_BOUND, UPPER_BOUND)

    generation_counter = 0 # Start a generation counter at 0
    while (GENERATIONS > generation_counter): # Termination condition. Can be set to (!SOLUTION_FOUND) to run until TARGET value is reached

        # Choose parents from the initial population based on roulette wheel probability selection
        # Will select amount of parents to satisfy the 'CROSSOVER_RATE'
        # If 'multi_selection' set to false, parents can only be chosen once each
        parents = selection_roulette(population, sum_squares_compute_fitness(population, TARGET), CROSSOVER_RATE, multi_selection=True)

        # Complete crossover of parents to produce their offspring
        # 'single_point_crossover' will choose 1 random position in each parents genome to crossover at
        children = single_point_crossover(parents)

        # Mutate the children using a random gene with random value with LOWER_BOUND < x < UPPER_BOUND range
        # The chance a child will be mutated is specified using 'MUTATION_RATE'
        # The amount of genes to mutate is specified using 'MUTATIONS'
        children = uniform_mutation(children, LOWER_BOUND, UPPER_BOUND, MUTATION_RATE, MUTATIONS)
        population = np.vstack((population, children)) # Add the mutated children back into the population

        # Calculate the next generation of the population, this is done by killing all the weakest individuals
        # until the population is reduced to 'POPULATION_SIZE'
        population = next_generation(population, sum_squares_compute_fitness(population, TARGET), POPULATION_SIZE)

        # Check if a solution is found
        # NOTE: this section of code can be commented out if using generational termination condition to increase optimisation
        if TARGET in sum_squares_compute_fitness(population, TARGET):
            TARGET = True

        # Increment the generation counter before reiterating through loop
        generation_counter += 1
        continue

    print('')
    print('   ##########################################################################################################################################')
    print('   ############################################################ FINAL GENERATION ############################################################')
    print('   ##########################################################################################################################################')
    display_population(population, sum_squares_compute_fitness(population, TARGET), population.shape[0])
    print('')
    print('FITTEST INDIVIDUAL')
    print('#############################')
    display_fittest_individual(population, sum_squares_compute_fitness(population, TARGET))
    print('#############################')

if __name__ == '__main__':
    main()
    
    
    
