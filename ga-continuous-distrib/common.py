import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def single_point_crossover_opt(parents):
    children = np.zeros_like(parents)  # Generate an empty array for the children the same shape as 'parents' array
    crossover_point = random.randrange(1, children.shape[2])  # Get a random index within the individuals index range
    # First child takes values from first parent up to crossover point, then the rest of the values from second parent
    # Second child takes values from second parent up to crossover point, then the rest of the values from first parent
    # This is done for each set of parents in 'parents' 3D array
    # This is done using NumPy horizontal stack: https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
    children[:, 0, :] = np.hstack([parents[:, 0, :crossover_point], parents[:, 1, crossover_point:]])
    children[:, 1, :] = np.hstack([parents[:, 1, :crossover_point], parents[:, 0, crossover_point:]])
    # Reshape children array vertically as they no longer need to be in pairs
    children = np.reshape(children, (children.shape[0] * children.shape[1], children.shape[2]))

    return children

def single_point_crossover_multi_index(parents):
    children = np.zeros_like(parents)  # Generate an empty array for the children the same shape as 'parents' array
    crossover_points = np.random.choice(range(1, children.shape[2]),  children.shape[0]) # Get a set of random indexes within the individuals index range
    # First child takes values from first parent up to crossover point, then the rest of the values from second parent
    # Second child takes values from second parent up to crossover point, then the rest of the values from first parent
    # This is done for each set of parents in 'parents' 3D array, with a different crossover point for each
    # This is done using NumPy horizontal stack: https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
    for i in range(children.shape[0]):
        children[i, 0, :] = np.hstack([parents[i, 0, :crossover_points[i]], parents[i, 1, crossover_points[i]:]])
        children[i, 1, :] = np.hstack([parents[i, 1, :crossover_points[i]], parents[i, 0, crossover_points[i]:]])
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

def non_uniform_mutation(children, fitness, lower_bound, upper_bound, mutation_rate, mutations):
    # Select indexes from the 'children' array based on the mutation_rate, which will be mutated
    children_selection = np.random.choice(a=[True, False], size=children.shape[0], p=[mutation_rate, 1 - mutation_rate])
    children_to_mutate = children[children_selection, :]
    # Randomly choose indexes for each 'children_to_mutate' (which correspond to individuals 'genes') within the individuals range,
    # for the given amount of 'mutations'
    mutation_indexes = np.random.choice(a=children_to_mutate.shape[1], size=(children_to_mutate.shape[0], mutations))
    # If the average fitness is greater than 1, replace the chosen indexes with random real values within the specified bounds,
    # else replace with a proportional random value based of the average fitness,
    # for each child selected for mutation np.mean(generation_fitness)
    if np.mean(fitness) > 1:
        mutated_genes = np.random.uniform(lower_bound, upper_bound, (children_to_mutate.shape[0], mutations))
        children_to_mutate[np.arange(children_to_mutate.shape[0])[:, None], mutation_indexes] = mutated_genes
    else:
        # Calculate new upper and lower bound based on: gene_to_mutate +- avg(fitness) * gene_to_mutate
        lower_bound = children_to_mutate[np.arange(children_to_mutate.shape[0])[:, None], mutation_indexes] \
                      - np.abs(np.mean(fitness) * children_to_mutate[np.arange(children_to_mutate.shape[0])[:, None], mutation_indexes])
        upper_bound =  children_to_mutate[np.arange(children_to_mutate.shape[0])[:, None], mutation_indexes] \
                       + np.abs(np.mean(fitness) * children_to_mutate[np.arange(children_to_mutate.shape[0])[:, None], mutation_indexes])

        mutated_genes = np.random.uniform(lower_bound, upper_bound, (children_to_mutate.shape[0], mutations))
        children_to_mutate[np.arange(children_to_mutate.shape[0])[:, None], mutation_indexes] = mutated_genes

    # Add the mutated children back into the 'children' array before returning them
    children = np.vstack((children, children_to_mutate))

    return children

def boundary_mutation(children, lower_bound, upper_bound, mutation_rate, mutations):
    # Select indexes from the 'children' array based on the mutation_rate, which will be mutated
    children_selection = np.random.choice(a=[True, False], size=children.shape[0], p=[mutation_rate, 1 - mutation_rate])
    children_to_mutate = children[children_selection, :]
    # Randomly choose indexes for each 'children_to_mutate' (which correspond to individuals 'genes') within the individuals range,
    # for the given amount of 'mutations'
    mutation_indexes = np.random.choice(a=children_to_mutate.shape[1], size=(children_to_mutate.shape[0], mutations))
    # Replace the chosen indexes with either the upper or lower bound, for each child selected for mutation
    mutated_genes = np.random.choice(a=[lower_bound, upper_bound], size=(children_to_mutate.shape[0], mutations))
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


def display_population(population, fitness, max_rows):
    # Generate a dataframe using the population individual values as first column, and their respective fitness as second column
    output_table = pd.DataFrame({'Individuals': population.tolist(),
                                 'Fitness':  fitness.tolist()})
    # Options for pandas set so that all data is visible before printing dataframe
    with pd.option_context('display.max_colwidth', -1, 'display.max_rows', max_rows):
        pd.set_option('colheader_justify', 'center')
        print(output_table)


def display_fittest_individual(population, fitness):
    # Generate a dataframe using the fittest individual in the population with formatting
    fittest_individual = pd.DataFrame({'Individual': population[np.argmin(fitness)]})
    fittest_individual['Individual'] = '      ' + fittest_individual['Individual'].astype(str)
    # Formatting options for pandas before printing data frame
    with pd.option_context('display.max_colwidth', -1):
        pd.set_option('colheader_justify', 'center')
        print(fittest_individual)
        print('-----------------------------')
        print('Fitness: ' + fitness[np.argmin(fitness)].astype(str))


def generate_plot(title, fitness_data):
    # Set axis labels and title
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fittest Individual")
    # Plot each thread with a different random colour, and annotate its final gen best fitness value to the left of chart
    for n in range(len(fitness_data)):
        plt.plot(range(len(fitness_data[n])), fitness_data[n], label='Thread ' + str(n), color=np.random.rand(3))
        plt.annotate('%0.8f' % fitness_data[n][-1], xy=(1, fitness_data[n][-1]), xytext=(8, 0),
                     xycoords=('axes fraction', 'data'), textcoords='offset points')
    # Thread legend set to above the graph, with max columns set to 5
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)


def plot_data_full(title, fitness_values):
    # Plot the fitness values against generations for the whole Y range
    generate_plot(title, fitness_values)
    plt.show()


def plot_data_ylim(title, fitness_values, ylim):
    # Plot the fitness values against generations for a specified Y range limit
    plt.ylim([0, ylim])
    generate_plot(title, fitness_values)
    plt.show()