- following readme template https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md

## Decentralized Multi-Armed Bandit Can Outperform Classic Upper Confidence Bound (NeurIPS 2021 Submission)

Official implementation of the Dec_UCB algorithm (Dec_UCB.py), and wrapper code (Dec_UCB_Simulator.py) used to run simulations found in the paper and compare to the classic UCB1 algorithm (UCB1.py).

### Requirements

A requirements.txt file has been provided for quick setup of dependencies. To use this file (Anaconda environment is used as an example), navigate in a terminal to the directory containing requirements.txt and execute the following commands, replacing `<env_name>` with the name of your new virtual environment:

`$ conda create -n <env_name> python=3.7`

`$ conda activate <env_name>`

`$ pip install -r requirements.txt`

If this does not work, or you would prefer not to go the virtual environment route, just be aware that to run the code you need Python 3 (development was completed with Python 3.7+) installed along with the following packages:
- networkx
- numpy
- scipy
- matplotlib

### Usage

### Reproducing Simulations

Small
python Dec_UCB_Simulator.py strong 6 heterogeneous -d bernoulli beta truncnorm --refreshMeans -f strongly.adjlist

python Dec_UCB_Simulator.py undirected 6 heterogeneous -d bernoulli beta truncnorm --refreshMeans -f undirected.adjlist

python Dec_UCB_Simulator.py weak 6 heterogeneous -d bernoulli beta truncnorm --refreshMeans -f weakly.adjlist

Large
python Dec_UCB_Simulator.py strong 10 heterogeneous -d bernoulli beta truncnorm --refreshMeans --refreshGraph -N 50

python Dec_UCB_Simulator.py undirected 10 heterogeneous -d bernoulli beta truncnorm --refreshMeans --refreshGraph -N 50

python Dec_UCB_Simulator.py weak 10 heterogeneous -d bernoulli beta truncnorm --refreshMeans --refreshGraph -N 50

One Distribution:
python Dec_UCB_Simulator.py strong 6 homogeneous -d beta --refreshMeans -f onedist-strong.adjlist

python Dec_UCB_Simulator.py undirected 6 homogeneous -d beta --refreshMeans -f onedist-undirected.adjlist

python Dec_UCB_Simulator.py weak 6 homogeneous -d beta --refreshMeans -f onedist-weak.adjlist

Three Distribution:
python Dec_UCB_Simulator.py strong 10 homogeneous -d bernoulli uniform truncnorm -s 0.2 --refreshMeans --refreshGraph -N 15

python Dec_UCB_Simulator.py strong 10 heterogeneous -d bernoulli uniform truncnorm -s 0.2 --refreshMeans --refreshGraph -N 15

python Dec_UCB_Simulator.py undirected 10 homogeneous -d bernoulli uniform truncnorm -s 0.2 --refreshMeans --refreshGraph -N 15

python Dec_UCB_Simulator.py undirected 10 heterogeneous -d bernoulli uniform truncnorm -s 0.2 --refreshMeans --refreshGraph -N 15

python Dec_UCB_Simulator.py weak 10 homogeneous -d bernoulli uniform truncnorm -s 0.2 --refreshMeans --refreshGraph -N 15

python Dec_UCB_Simulator.py weak 10 homogeneous -d bernoulli uniform truncnorm -s 0.2 --refreshMeans --refreshGraph -N 15
