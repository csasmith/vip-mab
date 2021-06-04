## Decentralized Multi-Armed Bandit Can Outperform Classic Upper Confidence Bound (NeurIPS 2021 Submission)

Official implementation of the Dec_UCB algorithm, and code used to generate simulations found in the paper.

### Requirements

A requirements.txt file has been provided for quick setup of dependencies. To use this file (Anaconda environment is used as an example), navigate in a terminal to the directory containing requirements.txt and execute the following commands, replacing `<env_name>` with the name of your new virtual environment:

`$ conda create -n <env_name> python=3.7`

`$ conda activate <env_name>`

`$ pip install -r requirements.txt`

If this does not work, or you would prefer not to go the virtual environment route, just be aware that to run the code you need Python 3 (development was completed with Python 3.7+) installed along with the following packages:
- [networkx](https://networkx.org/)
- [numpy](https://numpy.org/)
- [scipy](https://www.scipy.org/)
- [matplotlib](https://matplotlib.org/)

### Usage

`Dec_UCB_Simulator.py` is a script that runs simulations specified by command line arguments. This script was used to generate the simulations presented in the main paper and appendix. For example, to run a simulation on a size 4 weakly connected graph with 5 arms and homogeneous reward distributions from Bernoulli and uniform distributions, for 100 iterations, 1000 time steps each iteration, execute the following command:

`$ python Dec_UCB_Simulator.py weak 5 homogeneous -d bernoulli uniform -N 4`

Detailed usage instructions can be found by running `$ python Dec_UCB_Simulator.py --help`

In the event that the simulation options of `Dec_UCB_Simulator.py` are not flexible enough, `Dec_UCB.py` is the main implementation of the Dec_UCB algorithm that `Dec_UCB_Simulator.py` is simply a wrapper for. `Dec_UCB.py` handles fairly arbitrary inputs, so custom simulations can be easily written.

### Reproducing Simulations

Below is a list of all the commands necessary to reproduce the results found in the main paper and Appendix. Commands are in the order in which their corresponding figures appear in the paper.

**Small-size Graphs (Figures 1-3)**

`$ python Dec_UCB_Simulator.py strong 6 heterogeneous -d bernoulli beta truncnorm --refreshMeans -f strongly.adjlist`

`$ python Dec_UCB_Simulator.py undirected 6 heterogeneous -d bernoulli beta truncnorm --refreshMeans -f undirected.adjlist`

`$ python Dec_UCB_Simulator.py weak 6 heterogeneous -d bernoulli beta truncnorm --refreshMeans -f weakly.adjlist`

**Large-scale Graphs (Figure 4)**

`$ python Dec_UCB_Simulator.py strong 10 heterogeneous -d bernoulli beta truncnorm --refreshMeans --refreshGraph -N 50`

`$ python Dec_UCB_Simulator.py undirected 10 heterogeneous -d bernoulli beta truncnorm --refreshMeans --refreshGraph -N 50`

`$ python Dec_UCB_Simulator.py weak 10 heterogeneous -d bernoulli beta truncnorm --refreshMeans --refreshGraph -N 50`

**Homogeneous vs. Heterogeneous: One Distribution (Figures 5-7)**

`$ python Dec_UCB_Simulator.py strong 6 homogeneous -d beta -s 0.01 0.05 0.1 --refreshMeans --refreshGraph -N 6`

`$ python Dec_UCB_Simulator.py strong 6 heterogeneous -d beta -s 0.01 0.05 0.1 --refreshMeans --refreshGraph -N 6`

`$ python Dec_UCB_Simulator.py undirected 6 homogeneous -d beta -s 0.01 0.05 0.1 --refreshMeans --refreshGraph -N 6`

`$ python Dec_UCB_Simulator.py undirected 6 heterogeneous -d beta -s 0.01 0.05 0.1 --refreshMeans --refreshGraph -N 6`

`$ python Dec_UCB_Simulator.py weak 6 homogeneous -d beta -s 0.01 0.05 0.1 --refreshMeans --refreshGraph -N 6`

`$ python Dec_UCB_Simulator.py weak 6 heterogeneous -d beta -s 0.01 0.05 0.1 --refreshMeans --refreshGraph -N 6`

**Homogeneous vs. Heterogeneous: Three Distributions (Figures 8-10)**

`$ python Dec_UCB_Simulator.py strong 10 homogeneous -d bernoulli uniform truncnorm -s 0.2 --refreshMeans --refreshGraph -N 15`

`$ python Dec_UCB_Simulator.py strong 10 heterogeneous -d bernoulli uniform truncnorm -s 0.2 --refreshMeans --refreshGraph -N 15`

`$ python Dec_UCB_Simulator.py undirected 10 homogeneous -d bernoulli uniform truncnorm -s 0.2 --refreshMeans --refreshGraph -N 15`

`$ python Dec_UCB_Simulator.py undirected 10 heterogeneous -d bernoulli uniform truncnorm -s 0.2 --refreshMeans --refreshGraph -N 15`

`$ python Dec_UCB_Simulator.py weak 10 homogeneous -d bernoulli uniform truncnorm -s 0.2 --refreshMeans --refreshGraph -N 15`

`$ python Dec_UCB_Simulator.py weak 10 heterogeneous -d bernoulli uniform truncnorm -s 0.2 --refreshMeans --refreshGraph -N 15`

**Selected Graphs (Figures 11-14)**

`$ python Dec_UCB_Simulator.py undirected 6 heterogeneous -f undirected_two_neighbors.adjlist -d bernoulli truncnorm beta -m 0.10 0.25 0.45 0.65 0.75 0.90`

`$ python Dec_UCB_Simulator.py strong 6 heterogeneous -f strongly.adjlist -d bernoulli truncnorm beta -m 0.10 0.25 0.45 0.65 0.75 0.90`

`$ python Dec_UCB_Simulator.py undirected 6 heterogeneous -f undirected_path.adjlist -d bernoulli truncnorm beta -m 0.10 0.25 0.45 0.65 0.75 0.90`

`$ python Dec_UCB_Simulator.py weak 6 heterogeneous -f weakly_path.adjlist -d bernoulli truncnorm beta --refreshMeans`
