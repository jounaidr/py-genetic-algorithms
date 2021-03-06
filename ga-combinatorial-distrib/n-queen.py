from common import *
import numpy as np
import concurrent.futures
from threading import Lock
import time


THREADS = 5 # The amount of threads that will run the EA loop concurrently on the same population
print_lock = Lock() # Thread lock for the print statements

QUEENS = 100 # Used for individual size

POPULATION_SIZE = 100 # The maximum size of the population for each generation

CROSSOVER_RATE = 0.8 # The proportion of the population that will crossover to produce offspring each generation
MUTATION_RATE = 0.2 # The chance each offspring has of a gene (or multiple genes) being mutated each generation
MUTATIONS = 1 # The number of genes that are mutated if an offspring is selected for mutation (can be randomised with limits)

GENERATIONS = 10000 # The number of generations to run (if using as termination condition)
SOLUTION_FOUND = False # Whether an exact solution has been found (if using as termination condition)


def n_queen_compute_fitness(population):
    # Calculate the number of occurrences of each value in each individual in the population
    occurrences = np.apply_along_axis(np.bincount, axis=1, arr=population, minlength=np.max(population) + 1)
    # Sum the amount of occurrences generated previously
    count = np.apply_along_axis(np.bincount, axis=1, arr=occurrences, minlength=np.max(occurrences) + 1)
    # Subtract the sum of all single occurrences from the number of queens to get the fitness.
    # This value represents the how close each individual is to having a unique set of values (which corresponds to a queen on each row).
    fitness = QUEENS - count[:,1]

    return fitness


def main_threaded_loop(population, thread_no):
    global POPULATION_SIZE

    global GENERATIONS
    global SOLUTION_FOUND # Replace with local variable: SOLUTION_FOUND = False, to not stop other threads if solution is found in one thread

    global CROSSOVER_RATE
    global MUTATION_RATE
    global MUTATIONS

    thread_data = [0,[],[]] # List used to store execution time data at index 0, fittest value per gen at index 1 and mean fitness per gen at index 2

    # Calculate the fitness of the initial population and store fittest individual and mean fitness value data
    # NOTE: the following code can be commented out if data collection is not required
    initial_fitness = n_queen_compute_fitness(population)
    thread_data[1].append(initial_fitness[np.argmin(initial_fitness)])
    thread_data[2].append(np.mean(initial_fitness))

    # Start a generation counter at 1
    generation_counter = 1

    # Set the start time before EA loop
    start_time = time.time()

    # Termination condition. Can be set to just (SOLUTION_FOUND == False) to run until solution is found
    while (GENERATIONS > generation_counter) and (SOLUTION_FOUND == False):
        ###############################################################################
        ######################### EVOLUTIONARY ALGORITHM LOOP #########################
        ###############################################################################
        # Choose parents from the initial population based on roulette wheel probability selection
        # Will select amount of parents to satisfy the 'CROSSOVER_RATE'
        # If 'multi_selection' set to false, parents can only be chosen once each
        parents = selection_roulette(population, n_queen_compute_fitness(population), CROSSOVER_RATE, multi_selection=True)

        # Complete crossover of parents to produce their offspring
        # 'single_point_crossover' will choose 1 random position in each parents genome to crossover at
        children = single_point_crossover_opt(parents)

        # Mutate the children using integer mutation within the 'QUEENS' range
        # The chance a child will be mutated is specified using 'MUTATION_RATE'
        # The amount of genes to mutate is specified using 'MUTATIONS'
        children = integer_mutation(children, 0, QUEENS, MUTATION_RATE, MUTATIONS)
        population = np.vstack((population, children)) # Add the mutated children back into the population

        # Calculate the next generation of the population, this is done by killing all the weakest individuals
        # until the population is reduced to 'POPULATION_SIZE'
        population = next_generation(population, n_queen_compute_fitness(population), POPULATION_SIZE)
        ###############################################################################

        ###############################################################################
        ################################ DATA TRACKING ################################
        ###############################################################################
        # Calculate the fitness of the current gen population
        generation_fitness = n_queen_compute_fitness(population)

        # Store fittest individual and mean fitness value data
        # NOTE: this section can commented out if data collection is not required to increase optimisation
        thread_data[1].append(generation_fitness[np.argmin(generation_fitness)])
        thread_data[2].append(np.mean(generation_fitness))

        # Check if a solution is found
        if 0 in generation_fitness:
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
        display_population(population, n_queen_compute_fitness(population), population.shape[0])
        print('')
        print('FITTEST INDIVIDUAL:')
        print('')
        print('#############################')
        display_fittest_individual(population, n_queen_compute_fitness(population))
        # NOTE: PyPlot is NOT thread safe, running the below line can cause issues
        #plot_chessboard(population[np.argmin(n_queen_compute_fitness(population))], "THREAD " + str(thread_no))
        print('#############################')
        print('')
        print('EXECUTION TIME:')
        print(str(thread_data[0]) + 's')
        print('')

    return thread_data


if __name__ == '__main__':
    print('')
    print('#######################################################################################')
    print('########################### N-QUEENS EVOLUTIONARY ALGORITHM ###########################')
    print('#######################################################################################')

    # Generate initial population given parameters
    initial_population = generate_integer_population(POPULATION_SIZE, QUEENS)

    print('')
    print('INITIAL POPULATION:')
    display_population(initial_population, n_queen_compute_fitness(initial_population), initial_population.shape[0])
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


    plot_data_full("Fittest Individual", fittest_data)
    plot_data_full("Avg Fitness", avg_fitness_data)

    print('')
    print('#######################################################################################')
    print('################################ ALL THREADS EXECUTED! ################################')
    print('#######################################################################################')
    print('')

    generations_solution = []
    total_generations = 0

    for n in range(THREADS):
        print('THREAD: ' + str(n) + ' GENERATIONS: ' + str(len(fittest_data[n])), end="")
        total_generations += len(fittest_data[n])
        if 0 in fittest_data[n]:
            generations_solution.append(len(fittest_data[n]))
            print(', SOLUTION IN THREAD!')
        else:
            print()

    print('')
    print('MEAN EXECUTION TIME: ' + str(np.mean(execution_time_data)) + 's')
    print('MEAN GENERATIONS: ' + str(int(total_generations / THREADS)))
    #print('MEAN GENERATIONS UNTIL SOLUTION: ' + str(int(np.mean(generations_solution))))