import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import string


###############################################################################
####################### POPULATION GENERATION FUNCTIONS #######################
###############################################################################


def generate_binary_population(population_size, individual_size):
    # Generate empty 2D array of size population_size x individual_size
    population = np.zeros((population_size, individual_size))
    # Populate 'population' array with arrays of size individual_size of random binary digits
    for i in range(population_size):
        population[i,] = np.random.choice(a=[0, 1], size=individual_size)

    return population.astype(int) # Return population with integer values


def generate_string_population(population_size, individual_size):
    # Generate empty 2D array of size population_size x individual_size
    population = np.empty((population_size, individual_size), dtype=object)
    # Populate 'population' array with arrays of size individual_size of random ASCII chars (using python string module)
    for i in range(population_size):
        population[i,] = np.random.choice(a=list(string.printable), size=individual_size)

    return population

def generate_integer_population(population_size, individual_size):
    # Generate empty 2D array of size population_size x individual_size
    population = np.empty((population_size, individual_size), dtype=int)
    # Populate 'population' array with arrays of size individual_size of random integer values within 'individual_size' range
    for i in range(population_size):
        population[i,] = np.random.choice(a=individual_size, size=individual_size)

    return population


###############################################################################
############################# SELECTION FUNCTIONS #############################
###############################################################################


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


def selection_rank(population, fitness, crossover_rate, multi_selection=True):
    # Rank the fitness values, then inverse rankings to get minimum rankings, then calculate probabilities from the rankings
    rankings = (population.shape[0] + 1) - rankdata(fitness)
    probabilities = rankings[0:, ] / np.sum(rankings)
    # Calculate the number of parents to be selected based on crossover rate
    selection_amount = int((population.shape[0] * crossover_rate) / 2)
    # Select two individuals from the population given the weighted probabilities, for the amount specified by 'selection_amount'
    selection = np.random.choice(a=population.shape[0], replace=multi_selection, size=(selection_amount, 2),
                                 p=probabilities)
    parents = np.take(population, selection, axis=0)

    return parents


def selection_tournament(population, fitness, crossover_rate, tournament_proportion=0.3, multi_selection=True):
    # Calculate the number of participants in the tournament based on the tournament_proportion
    tournament_size = int(population.shape[0] * tournament_proportion)
    # Calculate the number of parents to be selected based on crossover rate
    selection_amount = int(population.shape[0] * crossover_rate)
    # Combine the population with their respective fitness values
    population_fitness = np.concatenate((population, fitness[:, None]), axis=1)
    # Select 'tournament_size' amount of individuals from the population, for the for the amount specified by 'selection_amount'
    tournament_selection = np.random.choice(a=population.shape[0], replace=multi_selection, size=(selection_amount, tournament_size))
    tournaments = np.take(population_fitness, tournament_selection, axis=0)
    # Get the winners from each tournament by using the the highest value individuals last index (fitness value) from each tournament,
    # and then remove the last element (the previously combined fitness value)
    winners = tournaments[np.arange(tournaments.shape[0]), np.argmax(tournaments[:, :, 10], axis=1), :-1]
    # Format the winners so they are in the (selection_amount/2 * 2 * individual_size) parents format before returning
    parents = np.transpose(np.asarray(np.vsplit(winners, 2)),  axes=[1, 0, 2])

    return parents


def selection_roulette_rank(population, fitness, crossover_rate, multi_selection=True, std_threshold=1):
    # If the standard deviation of fitness values is greater than 1, select using roulette wheel, otherwise use rank selection
    if np.std(fitness) > std_threshold:
        return selection_roulette(population, fitness, crossover_rate, multi_selection)
    else:
        return selection_rank(population, fitness, crossover_rate, multi_selection)


def selection_tournament_rank(population, fitness, crossover_rate, tournament_proportion=0.3, multi_selection=True, std_threshold=1):
    # If the standard deviation of fitness values is greater than 1, select using tournament selection, otherwise use rank selection
    if np.std(fitness) > std_threshold:
        return selection_tournament(population, fitness, crossover_rate, tournament_proportion, multi_selection)
    else:
        return selection_rank(population, fitness, crossover_rate, multi_selection)


###############################################################################
############################# CROSSOVER FUNCTIONS #############################
###############################################################################


def single_point_crossover_opt(parents):
    children = np.zeros_like(parents)  # Generate an empty array for the children the same shape as 'parents' array
    crossover_point = random.randrange(0, children.shape[2])  # Get a random index within the individuals index range
    # First child takes values from first parent up to crossover point, then the rest of the values from second parent
    # Second child takes values from second parent up to crossover point, then the rest of the values from first parent
    # This is done for each set of parents in 'parents' 3D array
    # This is done using NumPy horizontal stack: https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
    children[:, 0, :] = np.hstack([parents[:, 0, :crossover_point], parents[:, 1, crossover_point:]])
    children[:, 1, :] = np.hstack([parents[:, 1, :crossover_point], parents[:, 0, crossover_point:]])
    # Reshape children array vertically as they no longer need to be in pairs
    children = np.reshape(children, (children.shape[0] * children.shape[1], children.shape[2]))

    return children


def double_point_crossover_opt(parents):
    children = np.zeros_like(parents)  # Generate an empty array for the children the same shape as 'parents' array
    crossover_point_one = random.randrange(1, children.shape[2] - 1)  # Get the first random index within the individuals index range
    crossover_point_two = random.randrange(crossover_point_one, children.shape[2])  # Use the first crossover point for the bounds to generate the second
    # First child takes values from first parent up to crossover point one, then takes values from the second parent up until crossover point two, and the rest from first parent
    # Second child takes values from second parent up to crossover point one, then takes values from the first parent up until crossover point two, and the rest from second parent
    # This is done for each set of parents in 'parents' 3D array
    # This is done using NumPy horizontal stack: https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
    children[:, 0, :] = np.hstack([parents[:, 0, :crossover_point_one], parents[:, 1, crossover_point_one:crossover_point_two], parents[:, 0, crossover_point_two:]])
    children[:, 1, :] = np.hstack([parents[:, 1, :crossover_point_one], parents[:, 0, crossover_point_one:crossover_point_two], parents[:, 1, crossover_point_two:]])
    # Reshape children array vertically as they no longer need to be in pairs
    children = np.reshape(children, (children.shape[0] * children.shape[1], children.shape[2]))

    return children


def single_point_crossover_multi_index(parents):
    children = np.zeros_like(parents)  # Generate an empty array for the children the same shape as 'parents' array
    crossover_points = np.random.choice(range(1, children.shape[2]), children.shape[0]) # Get a set of random indexes within the individuals index range
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


###############################################################################
############################## MUTATION FUNCTIONS #############################
###############################################################################


def bit_flip_mutation(children, mutation_rate, mutations=1):
    # Select indexes from the 'children' array based on the mutation_rate, which will be mutated
    children_selection = np.random.choice(a=[True, False], size=children.shape[0], p=[mutation_rate, 1 - mutation_rate])
    children_to_mutate = children[children_selection, :]
    # Randomly choose indexes for each 'children_to_mutate' (which correspond to individuals 'genes') within the individuals range,
    # for the given amount of 'mutations'
    mutation_indexes = np.random.choice(a=children_to_mutate.shape[1], size=(children_to_mutate.shape[0], mutations))
    # Flip the bits for the chosen indexes, for each child selected for mutation
    mutated_genes = 1 - children_to_mutate[np.arange(children_to_mutate.shape[0])[:, None], mutation_indexes]
    children_to_mutate[np.arange(children_to_mutate.shape[0])[:, None], mutation_indexes] = mutated_genes
    # Add the mutated children back into the 'children' array before returning them
    children = np.vstack((children, children_to_mutate))

    return children


def string_mutation(children, mutation_rate, mutations=1):
    # Select indexes from the 'children' array based on the mutation_rate, which will be mutated
    children_selection = np.random.choice(a=[True, False], size=children.shape[0], p=[mutation_rate, 1 - mutation_rate])
    children_to_mutate = children[children_selection, :]
    # Randomly choose indexes for each 'children_to_mutate' (which correspond to individuals 'genes') within the individuals range,
    # for the given amount of 'mutations'
    mutation_indexes = np.random.choice(a=children_to_mutate.shape[1], size=(children_to_mutate.shape[0], mutations))
    # Replace the chosen indexes with random real values within the specified bounds, for each child selected for mutation
    mutated_genes = np.random.choice(a=list(string.printable), size=(children_to_mutate.shape[0], mutations))
    children_to_mutate[np.arange(children_to_mutate.shape[0])[:, None], mutation_indexes] = mutated_genes
    # Add the mutated children back into the 'children' array before returning them
    children = np.vstack((children, children_to_mutate))

    return children


def integer_mutation(children, lower_bound, upper_bound, mutation_rate, mutations=1):
    # Select indexes from the 'children' array based on the mutation_rate, which will be mutated
    children_selection = np.random.choice(a=[True, False], size=children.shape[0], p=[mutation_rate, 1 - mutation_rate])
    children_to_mutate = children[children_selection, :]
    # Randomly choose indexes for each 'children_to_mutate' (which correspond to individuals 'genes') within the individuals range,
    # for the given amount of 'mutations'
    mutation_indexes = np.random.choice(a=children_to_mutate.shape[1], size=(children_to_mutate.shape[0], mutations))
    # Replace the chosen indexes with random integer values within the specified bounds, for each child selected for mutation
    mutated_genes = np.random.randint(lower_bound, upper_bound, (children_to_mutate.shape[0], mutations))
    children_to_mutate[np.arange(children_to_mutate.shape[0])[:, None], mutation_indexes] = mutated_genes
    # Add the mutated children back into the 'children' array before returning them
    children = np.vstack((children, children_to_mutate))

    return children


###############################################################################
############################## SURVIVE FUNCTIONS ##############################
###############################################################################


def next_generation(previous_population, fitness, population_size):
    # Get indexes that correspond to the lowest fitness values for the amount specified by 'population_size'
    # This is done using NumPy 'argpartition' that has linear complexity: https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html
    fittest_indexes = np.argpartition(fitness, population_size - 1)[:population_size]
    # Generate the next generation from fittest individuals in the population specified by 'fittest_indexes'
    next_generation = previous_population[fittest_indexes]

    return next_generation


###############################################################################
########################### DATA PLOTTING FUNCTIONS ###########################
###############################################################################


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


def plot_chessboard(positions, title):
    # Source: https://stackoverflow.com/questions/23298961/how-to-create-a-certain-type-of-grid-with-matplotlib
    board = np.zeros((len(positions), len(positions), 3))
    board += 0.5  # "Black" color. Can also be a sequence of r,g,b with values 0-1.
    board[::2, ::2] = 1  # "White" color
    board[1::2, 1::2] = 1  # "White" color

    positions = positions

    fig, ax = plt.subplots()
    ax.imshow(board, interpolation='nearest')

    for x, y in enumerate(positions):
        # Use "family='font name'" to change the font
        ax.text(x, y, u'\u2655', size=30, ha='center', va='center')

    ax.set(xticks=[], yticks=[])
    ax.axis('image')
    plt.title(title)
    plt.show()