- following readme template https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md

## Decentralized Multi-Armed Bandit Can Outperform Classic Upper Confidence Bound (NeurIPS 2021 Submission)

Official implementation of the Dec_UCB algorithm (Dec_UCB.py), and wrapper code (Dec_UCB_Simulator.py) used to run simulations found in the paper and compare to the classic UCB1 algorithm (UCB1.py).

### Requirements

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
