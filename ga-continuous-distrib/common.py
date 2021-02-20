import random
import numpy as np

def generate_population(population_size, individual_size, lower_bound, upper_bound):
    # Generate empty 2D array of size population_size x individual_size
    population = np.zeros((population_size, individual_size))
    # Populate 'population' array with arrays of size individual_size of random float values between bounds
    for i in range(population_size):
        population[i,] = np.random.uniform(lower_bound, upper_bound, (individual_size))

    return population


def selection_roulette(population, fitness, crossover_rate, multi_selection=True):
    # Calculate each individuals fitness probability weighting using the total sum of each individuals fitness within the population
    probabilities = fitness[0:, ] / np.sum(fitness)
    # Calculate the number of parents to be selected based on crossover rate
    selection_amount = int((population.shape[0] * crossover_rate) / 2)
    # Select two individuals from the population given the weighted probabilities, for the amount specified by 'selection_amount'
    selection = np.random.choice(a=population.shape[0], replace=multi_selection, size=(selection_amount, 2),
                                 p=probabilities)
    parents = np.take(population, selection, axis=0)

    return parents


def single_point_crossover(parents):
    children = np.zeros_like(parents)  # Generate an empty array for the children the same shape as 'parents' array
    crossover_point = random.randrange(1, parents.shape[2])  # Get a random index within the individuals index range
    # First child takes values from first parent up to crossover point, then the rest of the values from second parent
    # Second child takes values from second parent up to crossover point, then the rest of the values from first parent
    # This is done for each set of parents in 'parents' 3D array
    # This is done using NumPy horizontal stack: https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
    children[:, 0, :] = np.hstack([parents[:, 0, :crossover_point], parents[:, 1, crossover_point:]])
    children[:, 1, :] = np.hstack([parents[:, 1, :crossover_point], parents[:, 0, crossover_point:]])
    # Reshape children array vertically as they no longer need to be in pairs
    children = np.reshape(children, (children.shape[0] * children.shape[1], children.shape[2]))

    return children


def uniform_mutation(children, lower_bound, upper_bound, mutation_rate, mutations):
    # Select indexes from the 'children' array based on the mutation_rate, which will be mutated
    children_selection = np.random.choice(a=[True, False], size=children.shape[0], p=[mutation_rate, 1 - mutation_rate])
    children_to_mutate = children[children_selection, :]
    # Randomly choose indexes for each 'children_to_mutate' (which correspond to individuals 'genes') within the individuals range,
    # for the given amount of 'mutations'
    mutation_indexes = np.random.choice(a=children_to_mutate.shape[1], size=(children_to_mutate.shape[0], mutations))
    # Replace the chosen indexes with random real values within the specified bounds, for each child selected for mutation
    mutated_genes = np.random.uniform(lower_bound, upper_bound, (children_to_mutate.shape[0], mutations))
    children_to_mutate[np.arange(children_to_mutate.shape[0])[:, None], mutation_indexes] = mutated_genes
    # Add the mutated children back into the 'children' array before returning them
    children = np.vstack((children, children_to_mutate))

    return children


def next_generation(previous_population, fitness, population_size):
    # Get indexes that correspond to the lowest fitness values for the amount specified by 'population_size'
    # This is done using NumPy 'argpartition' that has linear complexity: https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html
    fittest_indexes = np.argpartition(fitness, population_size - 1)[:population_size]
    # Generate the next generation from fittest individuals in the population specified by 'fittest_indexes'
    next_generation = previous_population[fittest_indexes]

    return next_generation