# Optimization algorithms

This repository contains the code for the project implementing various optimization algorithms (gradient descent, Gauss-Newton, Levenberg-Marquardt, stochastic gradient descent) and comparing them on a set of problems.

## Repository structure

* `optimizers.py` - contains the implementation of the optimization algorithms
* `functions.py` - a library for easy generation of functions with their derivatives as compositions of elementary functions
* `test_simple_function.ipynb` - a notebook with a simple function to test the optimizers
* `test_complicated_function.ipynb` - a notebook with a more complicated function to test the optimizers, the `functions.py` library is heavily used here
* `test_neural_network.ipynb` - a notebook with a neural network to test the optimizers