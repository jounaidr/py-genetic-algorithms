from common import *
import numpy as np
import concurrent.futures
from threading import Lock
import time


THREADS = 5 # The amount of threads that will run the EA loop concurrently on the same population
print_lock = Lock() # Thread lock for the print statements

ITEM_AMOUNT = 10 # Also used as individual size
ITEMS = np.zeros((ITEM_AMOUNT, 2))

KNAPSACK_MAX_WEIGHT = 35

POPULATION_SIZE = 100 # The maximum size of the population for each generation

CROSSOVER_RATE = 0.8 # The proportion of the population that will crossover to produce offspring each generation
MUTATION_RATE = 0.2 # The chance each offspring has of a gene (or multiple genes) being mutated each generation
MUTATIONS = 1 # The number of genes that are mutated if an offspring is selected for mutation (can be randomised with limits)

GENERATIONS = 100 # The number of generations to run (if using as termination condition)


def generate_items():
    # Generate items of random weight from 1 < x < 15 and value 10 < x < 1000
    ITEMS[:, 0] = np.random.randint(1, 15, size=ITEM_AMOUNT)
    ITEMS[:, 1] = np.random.randint(10, 1000, size = 10)
    # Print the items with a pandas dataframe
    print(pd.DataFrame({'Weight': ITEMS[:, 0],
                        'Value':  ITEMS[:, 1]}).astype(int))


def sum_knapsack_compute_fitness(population):
    # Generate boolean mask using population
    population_bool = np.array(population, dtype=bool)
    # Sum the 'value' values from ITEMS that satisfies the population boolean mask
    fitness = np.tile(ITEMS[:, 1], (population_bool.shape[0], 1)).astype(int)
    fitness[~population_bool] = 0
    fitness = np.sum(fitness, axis=1)
    # Sum the 'weight' values from ITEMS that satisfies the population boolean mask
    weight = np.tile(ITEMS[:, 0], (population_bool.shape[0], 1)).astype(int)
    weight[~population_bool] = 0
    weight = np.sum(weight, axis=1)
    # Replace fitness with 0 where if the fitness values respective total weight is greater than KNAPSACK_MAX_WEIGHT
    fitness[(weight >= KNAPSACK_MAX_WEIGHT)] = 0
    # Subtract fitness value from maximum value so fitness in minimized
    fitness = np.sum(ITEMS[:, 1]) - fitness

    return fitness


def main_threaded_loop(population, thread_no):
    global ITEM_AMOUNT
    global ITEMS

    global POPULATION_SIZE

    global GENERATIONS

    global CROSSOVER_RATE
    global MUTATION_RATE
    global MUTATIONS

    thread_data = [0,[],[]] # List used to store execution time data at index 0, fittest value per gen at index 1 and mean fitness per gen at index 2

    # Calculate the fitness of the initial population and store fittest individual and mean fitness value data
    # NOTE: the following code can be commented out if data collection is not required
    initial_fitness = np.sum(ITEMS[:, 1]) - sum_knapsack_compute_fitness(population)
    thread_data[1].append(initial_fitness[np.argmin(initial_fitness)])
    thread_data[2].append(np.mean(initial_fitness))

    # Start a generation counter at 1
    generation_counter = 1

    # Set the start time before EA loop
    start_time = time.time()

    # Termination condition.
    while (GENERATIONS > generation_counter):
        ###############################################################################
        ######################### EVOLUTIONARY ALGORITHM LOOP #########################
        ###############################################################################
        # Choose parents from the initial population based on roulette wheel probability selection
        # Will select amount of parents to satisfy the 'CROSSOVER_RATE'
        # If 'multi_selection' set to false, parents can only be chosen once each
        parents = selection_roulette(population, sum_knapsack_compute_fitness(population), CROSSOVER_RATE, multi_selection=True)

        # Complete crossover of parents to produce their offspring
        # 'single_point_crossover' will choose 1 random position in each parents genome to crossover at
        children = single_point_crossover_opt(parents)

        # Mutate the children using bit flip operation
        # The chance a child will be mutated is specified using 'MUTATION_RATE'
        # The amount of genes to mutate is specified using 'MUTATIONS'
        children = bit_flip_mutation(children, MUTATION_RATE, MUTATIONS)
        population = np.vstack((population, children)) # Add the mutated children back into the population

        # Calculate the next generation of the population, this is done by killing all the weakest individuals
        # until the population is reduced to 'POPULATION_SIZE'
        population = next_generation(population, sum_knapsack_compute_fitness(population), POPULATION_SIZE)
        ###############################################################################

        ###############################################################################
        ################################ DATA TRACKING ################################
        ###############################################################################
        # Calculate the fitness of the current gen population
        generation_fitness = np.sum(ITEMS[:, 1]) - sum_knapsack_compute_fitness(population)

        # Store fittest individual and mean fitness value data
        # NOTE: this section can commented out if data collection is not required to increase optimisation
        thread_data[1].append(generation_fitness[np.argmin(generation_fitness)])
        thread_data[2].append(np.mean(generation_fitness))

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
        print('FITTEST INDIVIDUAL:')
        print('')
        print('#############################')
        population_bool = np.array(population, dtype=bool)
        print(pd.DataFrame({'Weight': ITEMS[population_bool[np.argmax(sum_knapsack_compute_fitness(population)),:], 0],
                            'Value': ITEMS[population_bool[np.argmax(sum_knapsack_compute_fitness(population)),:], 1]}).astype(int))
        print('Total Value: ' + sum_knapsack_compute_fitness(population)[np.argmax(sum_knapsack_compute_fitness(population))].astype(str))
        print('#############################')
        print('')
        print('EXECUTION TIME:')
        print(str(thread_data[0]) + 's')
        print('')

    return thread_data

if __name__ == '__main__':
    print('')
    print('#######################################################################################')
    print('########################## SUM BINARY EVOLUTIONARY ALGORITHM ##########################')
    print('#######################################################################################')

    # Generate and print items
    print('')
    print('ITEMS:')
    generate_items()
    print('')

    # Generate initial population given parameters
    initial_population = generate_binary_population(POPULATION_SIZE, ITEM_AMOUNT)

    print('')
    print('STARTING EVOLUTIONARY ALGORITHM THREADS...')
    print('')

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
        total_generations += len(fittest_data[n])

    print('')
    print('MEAN EXECUTION TIME: ' + str(np.mean(execution_time_data)) + 's')
    print('MEAN GENERATIONS: ' + str(int(total_generations / THREADS)))
    #print('MEAN GENERATIONS UNTIL SOLUTION: ' + str(int(np.mean(generations_solution))))