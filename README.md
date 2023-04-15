# Learning to predict arbitrary quantum processes

This open-source implementation considers a machine learning (ML) algorithm for predicting the output properties of an arbitrarily complex quantum process (the quantum process could even be an exponentially large quantum circuit!).

We require `g++` (C++ compiler), `python` version 3, and Jupyter Notebook (https://ipython.org/notebook.html).

On the experimental side, we require **preparation of product states** and **single-qubit Pauli measurements** (i.e., each measurement measures all qubits on some Pauli X, Y, or Z- basis). This should be readily available in many quantum platforms.

An introduction to this ML algorithm and the underlying theory can be found in our papers: https://arxiv.org/abs/2210.14894

## Quick Start

Every folder (except for `Eigen/`), such as `50spins-allZ-many-t-homogeneous` or `Sys-40spins-oneZ-allt-homogeneous`, corresponds to a particular quantum system that we consider in the numerical experiments. The folder `Eigen/` is an open-source library (https://eigen.tuxfamily.org/index.php?title=Main_Page) for performing eigendecomposition.

To create the executable files in each folder (`XXZ` or `XXZ-more-general`), type `make` in each folder (this requires the C++ compiler `g++`). Running the executable file `./XXZ` creates `states.txt` and `values.txt`, which consist of the training data for the ML algorithm.

The training and prediction of the ML algorithm are given in `LearningQuantumProcess.ipynb` (this requires Jupyter Notebook). To open `LearningQuantumProcess.ipynb`, type `ipython notebook` in the main folder and click `LearningQuantumProcess.ipynb` on the webpage.
