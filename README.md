# Optimization algorithms

This repository contains the code for the project implementing various optimization algorithms (gradient descent, Newthon method, Gauss-Newton method, Levenberg-Marquardt method) and comparing them on simple and more complicated functions and a simple neural network.

## Repository structure

The repository contains two main folders:
* `src` - contains the source code for the project
* `tests` - contains notebooks with tests of the optimization algorithms

The `src` folder contains the following files:
* `optimizers.py` - contains the implementation of the optimization algorithms
* `functions.py` - a library for easy generation of functions with their first and second derivatives as compositions of elementary functions

The `tests` folder contains the following files:
* `polynomial.ipynb` - a notebook with a polynomial function to test the optimizers
* `simple_function.ipynb` - a notebook with a simple function to test the optimizers
* `complicated_function.ipynb` - a notebook with a more complicated function to test the optimizers, the `functions.py` library is heavily used here
* `neural_network.ipynb` - a notebook with a neural network to test the optimizers