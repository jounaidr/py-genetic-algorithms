from common import *
import numpy as np
import concurrent.futures
from threading import Lock
import time


THREADS = 5 # The amount of threads that will run the EA loop concurrently on the same population
print_lock = Lock() # Thread lock for the print statements

POPULATION_SIZE = 10 # The maximum size of the population for each generation

INDIVIDUAL_SIZE = 10 # The number genes in each individual in the population, global minima at: 2,5,10
LOWER_BOUND = 0 # The lower limit that a gene value can be, default = 0
UPPER_BOUND = np.pi # The upper limit that a gene value can be, default = pi

TARGET = -9.66015 # for INDIVIDUAL_SIZE = 2,5,10, TARGET =  -1.8013, -4.687658, -9.66015 respectively
STEEPNESS = 10 # Steepness of valleys and ridges, larger value increases search difficulty, default = 10

CROSSOVER_RATE = 0.8 # The proportion of the population that will crossover to produce offspring each generation
MUTATION_RATE = 0.2 # The chance each offspring has of a gene (or multiple genes) being mutated each generation
MUTATIONS = 1 # The number of genes that are mutated if an offspring is selected for mutation (can be randomised with limits)

GENERATIONS = 10000 # The number of generations to run (if using as termination condition)
SOLUTION_FOUND = False # Whether an exact solution has been found (if using as termination condition)


def michalewicz_compute_fitness(population):
    # Generate a 1D array of indexes from 1 to the individuals size
    i = np.arange(1, population.shape[1] + 1)
    # Calculate the result based on: -sum(sin(x) * (sin((ix^2) / pi)^2*STEEPNESS)), for each individuals values in the population
    result = -np.sum(np.sin(population[0:,]) * (np.sin((i * population[0:,] ** 2) / np.pi) ** (2 * STEEPNESS)), axis=1)
    fitness = abs(result[0:,] - TARGET) # Calculate the results absolute distance from target, the minimal solution

    return fitness


def main_threaded_loop(population, thread_no):
    global POPULATION_SIZE
    global INDIVIDUAL_SIZE

    global LOWER_BOUND
    global UPPER_BOUND

    global GENERATIONS
    global SOLUTION_FOUND

    global CROSSOVER_RATE
    global MUTATION_RATE
    global MUTATIONS

    thread_data = [0,[],[]] # List used to store execution time data at index 0, fittest value per gen at index 1 and mean fitness per gen at index 2

    # Calculate the fitness of the initial population and store fittest individual and mean fitness value data
    # NOTE: the following code can be commented out if data collection is not required
    initial_fitness = michalewicz_compute_fitness(population)
    thread_data[1].append(initial_fitness[np.argmin(initial_fitness)])
    thread_data[2].append(np.mean(initial_fitness))

    # Start a generation counter at 1
    generation_counter = 1

    # Set the start time before EA loop
    start_time = time.time()

    # Termination condition. Can be set to just (SOLUTION_FOUND == True) to run until solution is found
    while (GENERATIONS > generation_counter) or (SOLUTION_FOUND == True):
        ###############################################################################
        ######################### EVOLUTIONARY ALGORITHM LOOP #########################
        ###############################################################################
        # Choose parents from the initial population based on roulette wheel probability selection
        # Will select amount of parents to satisfy the 'CROSSOVER_RATE'
        # If 'multi_selection' set to false, parents can only be chosen once each
        parents = selection_roulette(population, michalewicz_compute_fitness(population), CROSSOVER_RATE, multi_selection=True)

        # Complete crossover of parents to produce their offspring
        # 'single_point_crossover' will choose 1 random position in each parents genome to crossover at
        children = single_point_crossover_opt(parents)

        # Mutate the children using a random gene with random value with LOWER_BOUND < x < UPPER_BOUND range
        # The chance a child will be mutated is specified using 'MUTATION_RATE'
        # The amount of genes to mutate is specified using 'MUTATIONS'
        children = uniform_mutation(children, LOWER_BOUND, UPPER_BOUND, MUTATION_RATE, MUTATIONS)
        population = np.vstack((population, children)) # Add the mutated children back into the population

        # Calculate the next generation of the population, this is done by killing all the weakest individuals
        # until the population is reduced to 'POPULATION_SIZE'
        population = next_generation(population, michalewicz_compute_fitness(population), POPULATION_SIZE)
        ###############################################################################

        ###############################################################################
        ################################ DATA TRACKING ################################
        ###############################################################################
        # Calculate the fitness of the current gen population
        generation_fitness = michalewicz_compute_fitness(population)

        # Store fittest individual and mean fitness value data
        # NOTE: this section can commented out if data collection is not required to increase optimisation
        thread_data[1].append(generation_fitness[np.argmin(generation_fitness)])
        thread_data[2].append(np.mean(generation_fitness))

        # Check if a solution is found
        if (INDIVIDUAL_SIZE == 2) and (-1.8013 in generation_fitness):
            SOLUTION_FOUND = True
        if (INDIVIDUAL_SIZE == 5) and (-4.687658 in generation_fitness):
            SOLUTION_FOUND = True
        if (INDIVIDUAL_SIZE == 10) and (-9.66015 in generation_fitness):
            SOLUTION_FOUND = True

        # Increment the generation counter before reiterating through loop
        generation_counter += 1
        ###############################################################################
        continue

    # Calculate the EA loops execution time and store data
    thread_data[0] = time.time() - start_time

    # After termination condition is met, lock thread and print results before returning data
    with print_lock:
        print('')
        print('##################################################################################################################################')
        print('############################################################ THREAD ' + str(thread_no) + ' ############################################################')
        print('##################################################################################################################################')
        print('')
        print('EXECUTION TIME:')
        print('')
        print(str(thread_data[0]) + 's')
        print('')
        print('FINAL GENERATION:')
        display_population(population, michalewicz_compute_fitness(population), population.shape[0])
        print('')
        print('FITTEST INDIVIDUAL:')
        print('')
        print('#############################')
        display_fittest_individual(population, michalewicz_compute_fitness(population))
        print('#############################')
        print('')
        print('EXECUTION TIME:')
        print(str(thread_data[0]) + 's')
        print('')

    return thread_data

if __name__ == '__main__':
    print('')
    print('#######################################################################################')
    print('######################### MICHALEWICZ EVOLUTIONARY ALGORITHM ##########################')
    print('#######################################################################################')

    # Generate initial population given parameters
    initial_population = generate_population(POPULATION_SIZE, INDIVIDUAL_SIZE, LOWER_BOUND, UPPER_BOUND)

    print('')
    print('INITIAL POPULATION:')
    display_population(initial_population, michalewicz_compute_fitness(initial_population), initial_population.shape[0])
    print('')
    print('STARTING EVOLUTIONARY ALGORITHM THREADS...')

    data = [] # Initialise list to store thread_data futures
    # Initialise a ThreadPoolExecutor with 'THREADS' thread pool size
    # and execute the 'main_threaded_loop' on each thread in the pool
    # store the return futures for each thread to be processed later...
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
        for n in range(THREADS):
            data.append(executor.submit(main_threaded_loop, initial_population, n))


    execution_time_data = [] # EA loop execution time in seconds
    fittest_data = [] # Fittest individual in each generation
    avg_fitness_data = [] # Average (mean) fitness of each generation

    # Unpack the 'data' futures list data for each thread and store in a separate list for each data set
    for n in range(THREADS):
        execution_time_data.append(data[n].result()[0])
        fittest_data.append(data[n].result()[1])
        avg_fitness_data.append(data[n].result()[2])

    # Plot fittest individual against generations for full fitness range, then from 0 < x < 1 fitness range
    plot_data_full("Fittest Individual Full", GENERATIONS, fittest_data)
    plot_data_ylim("Fittest Individual Limited", GENERATIONS, fittest_data, 1)
    # Plot average fitness against generations for full fitness range, then from 0 < x < 1 fitness range
    plot_data_full("Avg Fitness Full", GENERATIONS, avg_fitness_data)
    plot_data_ylim("Avg Fitness Limited", GENERATIONS, avg_fitness_data, 1)

    print('')
    print('#######################################################################################')
    print('################################ ALL THREADS EXECUTED! ################################')
    print('#######################################################################################')
    print('')
    print('MEAN EXECUTION TIME: ' + str(np.mean(execution_time_data)) + 's')
    print('SOLUTION FOUND: ' + str(SOLUTION_FOUND))