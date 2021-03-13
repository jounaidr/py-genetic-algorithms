# py-genetic-algorithms
Sandbox repo for various genetic algorithms, writing in python.

### How to run scripts

There are two packages containing problem scripts for both [continuous](https://github.com/jounaidr/py-genetic-algorithms/tree/master/ga-continuous-distrib) and [combinatorial](https://github.com/jounaidr/py-genetic-algorithms/tree/master/ga-combinatorial-distrib) optimisation.\
Each package contains a `common.py` script for the: population generation; selection; crossover; mutation; survive and data visualisation methods can be found (each section seperated by comment headers).\
Each problem has a respective script which imports the common methods for use within the `main_threaded_loop` function (which execute a GA run for each thread defined by the `THREADS` global param).\
Each problem uses a different fitness function defined within each script respectively, with some problems requiring slightly different implementation.\
The global parameters for each script are set their respective default value, which can be changed by directly modifying the values within each script.\
Each scripts `main_threaded_loop` uses the 'basic' selection, mutation and crossover methods by default, which can be changed by directly modifying the script to the desired methods.\
Each script can be directly run from console, however the scripts for `ga-continuous-distrib` and `ga-continuous-distrib` must be in separate packages as each package uses a separate `common.py` script.\ 

#### ga-continuous-distribution

The following continuous optimisation problems are implemented:
- [Sum Squares](https://github.com/jounaidr/py-genetic-algorithms/blob/master/ga-continuous-distrib/sum-squares.py)
- [Schwefel](https://github.com/jounaidr/py-genetic-algorithms/blob/master/ga-continuous-distrib/schwefel.py)
- [Michalewicz](https://github.com/jounaidr/py-genetic-algorithms/blob/master/ga-continuous-distrib/michalewicz.py)
- [Matyas](https://github.com/jounaidr/py-genetic-algorithms/blob/master/ga-continuous-distrib/matyas.py)
- [Bukin N6](https://github.com/jounaidr/py-genetic-algorithms/blob/master/ga-continuous-distrib/bukin-6.py)
- [Six Hump Camel](https://github.com/jounaidr/py-genetic-algorithms/blob/master/ga-continuous-distrib/six-hump-camel.py)

#### ga-combinatorial-distribution

The following combinatorial optimisation problems are implemented:
- [Sum Binary](https://github.com/jounaidr/py-genetic-algorithms/blob/master/ga-combinatorial-distrib/sum-binary.py)
- [Partition](https://github.com/jounaidr/py-genetic-algorithms/blob/master/ga-combinatorial-distrib/partition.py)
- [Phrase Match](https://github.com/jounaidr/py-genetic-algorithms/blob/master/ga-combinatorial-distrib/phrase-match.py)
- [N Queens](https://github.com/jounaidr/py-genetic-algorithms/blob/master/ga-combinatorial-distrib/n-queen.py)
- [Knapsack](https://github.com/jounaidr/py-genetic-algorithms/blob/master/ga-combinatorial-distrib/knapsack.py)
