''' Perform desired Dec_UCB simulations from standard input '''

from Dec_UCB import Dec_UCB
from UCB1 import UCB1
import argparse
import networkx as nx
import numpy as np
import scipy.stats as sps
import random
import matplotlib.pyplot as plt

def generate_random_graph(size, type):
    if type == 'undirected':
        G = nx.fast_gnp_random_graph(size, 0.5, directed=False)
        while not nx.is_connected(G):
            G = nx.fast_gnp_random_graph(size, 0.5, directed=False)
    else:
        G = nx.fast_gnp_random_graph(size, 0.5, directed=True)
        if type == 'strong':
            while not nx.is_strongly_connected(G):
                G = nx.fast_gnp_random_graph(size, 0.5, directed=True)
        else:
            while nx.is_strongly_connected(G) or not nx.is_weakly_connected(G):
                G = nx.fast_gnp_random_graph(size, 0.5, directed=True)
    # add self-loops
    nodes = list(G.nodes)
    for i in nodes:
        G.add_edge(i,i) 
    return G

# parse arguments from standard input
parser = argparse.ArgumentParser()
group1 = parser.add_mutually_exclusive_group()
group1.add_argument('-N', '--numAgents', type=int, default=5, help="Number of agents")
group1.add_argument('-f', '--inputFile', help="text file path containing a NetworkX Graph " + 
        "or DiGraph in multiline adjacency list format. The graph must be nonempty and each node must " + 
        "have a self-loop. Additionally, the graph must match the type parameter")
parser.add_argument('type', choices=['strong', 'weak', 'undirected'], help="Graph type must " + 
        "be either strongly connected, weakly connected, or undirected connected")
parser.add_argument('numArms', type=int, default='6', help="Number of arms")
parser.add_argument('setting', choices=['homogeneous', 'heterogeneous'], help="Arm distributions can " + 
        "be homogeneous or heterogeneous")
parser.add_argument('-m', '--means', type=float, nargs='+', help="List of M arm means within (0,1)")
parser.add_argument('-d', '--distributions', nargs='+',
        choices=['truncnorm', 'beta', 'bernoulli', 'uniform'],
        default=['truncnorm'], help="List of scipy probability distribution names")
parser.add_argument('-s', '--stddevs', type=float, nargs='+', default=[0.05],
        help="List of standard deviations, if applicable to any distribution. If a standard deviation " +  
        "is incompatible with the corresponding mean for a beta distribution, a valid standard deviation " + 
        "is generated. Standard deviations are randomly assigned to arms in the homogeneous case, and " + 
        "randomly assigned to agent/arm pairs in the heterogeneous case")
parser.add_argument('-t', '--time', type=int, default=1000, help="Number of time steps")
parser.add_argument('-e', '--epochs', type=int, default=100, help="Number of iterations " +  
        "that Dec_UCB is repeated for")
parser.add_argument('--refreshMeans', action='store_true', help='If specified, a new set of arm_means ' +
        'is generated every epoch')
parser.add_argument('--refreshGraph', action='store_true', help='If specified, a new random graph is ' +
        'generated every epoch')
args = parser.parse_args()

# if no means provided, have to generate defaults here once numArms is known
if args.means == None:
    args.means=[random.uniform(0.05, 0.95) for x in range(args.numArms)]

# additional validation - any validation that appears to be missing is in Dec_UCB.py
if args.inputFile:
    if args.type == 'undirected':
        G = nx.read_multiline_adjlist(args.inputFile)
    else:
        G = nx.read_multiline_adjlist(args.inputFile, create_using=nx.DiGraph)
    if args.type == 'strong' and not nx.is_strongly_connected(G):
        raise TypeError("Graph type must match type argument")
    if args.type == 'weak' and (not nx.is_weakly_connected(G) or nx.is_strongly_connected(G)):
        raise TypeError("Graph type must match type argument")
    if args.type == 'undirected' and not nx.is_connected(G):
        raise TypeError("Graph type must match type argument")
if args.numAgents and args.numAgents <= 0:
    raise ValueError("numAgents needs to be a positive integer")
if args.numArms <= 0:
    raise ValueError("numArms needs to be a positive integer")
if len(args.means) != args.numArms:
    raise ValueError("means needs to be a list of numArms floats between 0 and 1")
if any(s < 0 for s in args.stddevs):
    raise ValueError("standard deviation needs to be a non-negative float")
if args.time <= 0:
    raise ValueError('The number of time steps must be a positive integer')
if args.epochs <= 0:
    raise ValueError("The number of epochs must be a positive integer")

# if file input provided, assume we don't want to refresh graph every epoch
args.refreshGraph = False if args.inputFile else args.refreshGraph

# randomly generate graph if -f option not used
if args.numAgents and args.inputFile == None:
    G = generate_random_graph(args.numAgents, args.type)
numAgents = G.number_of_nodes()

# get opcode from graph type
if args.type == 'strong' or args.type == 'weak' or numAgents == 1:
    opcode = 1 # note if N=1 we do not want undirected weights so opcode = 1
else:
    opcode = 2

def set_distribution(d, mean, sd):
    if d == 'truncnorm':
        a = (0 - mean) / sd
        b = (1 - mean) / sd
        return sps.truncnorm(a, b, loc=mean, scale=sd)
    if d == 'bernoulli':
        return sps.bernoulli(mean)
    if d == 'beta':
        if sd**2 < mean * (1 - mean):
            var = sd**2
        else:
            var = mean * (1 - mean) * 0.999 # var < mu(1 - mu) inequality for beta distributions
            print('Std dev ' + str(sd) + ' incompatible. Generated new std dev ' + str(np.sqrt(var)))
        alpha = mean * (mean * (1 - mean) / var - 1)
        beta = (1 - mean) * (mean * (1 - mean) / var - 1)
        return sps.beta(alpha, beta)
    if d == 'uniform':
        # we wish to obtain a uniform distribution given a certain mean, and we do this by
        # picking the widest uniform distribution possible still in [0,1] with the given mean
        radius = min(mean, abs(1 - mean))
        return sps.uniform(loc=mean - radius, scale=2*radius)

def generate_distributions(setting, numArms, numAgents, distributionOptions, means, stddevs):
    distributions = [[None for i in range(numArms)] for i in range(numAgents)]
    if setting == 'heterogeneous':
        for i in range(numAgents):
            for j in range(numArms):
                d = random.choice(distributionOptions)
                sd = random.choice(stddevs)
                distributions[i][j] = set_distribution(d, means[j], sd)
    else:
        for j in range(numArms):
            d = random.choice(distributionOptions)
            sd = random.choice(stddevs)
            for i in range(numAgents):
                distributions[i][j] = set_distribution(d, means[j], sd)
    return distributions

distributions = generate_distributions(args.setting, args.numArms, numAgents, args.distributions, args.means, args.stddevs)

# run simulations
regrets_Dec_UCB = []
regrets_UCB1 = []
means = args.means
simulator_Dec_UCB = Dec_UCB(G, args.time, opcode, means, distributions)
simulator_UCB1 = UCB1(args.time, means, distributions, numAgents)
for e in range(args.epochs):
    regrets_Dec_UCB.append(simulator_Dec_UCB.run())
    regrets_UCB1.append(simulator_UCB1.run())
    means = [random.uniform(0.05, 0.95) for x in range(args.numArms)] if args.refreshMeans else means
    distributions = generate_distributions(args.setting, args.numArms, numAgents, args.distributions, means, args.stddevs)
    G = generate_random_graph(numAgents, args.type) if args.refreshGraph else G
    simulator_Dec_UCB = Dec_UCB(G, args.time, opcode, means, distributions)
    simulator_UCB1 = UCB1(args.time, means, distributions, numAgents)
    print('epoch: ' + str(e)) if e % 10 == 0 else None
regrets_Dec_UCB = np.asarray(regrets_Dec_UCB)
regrets_UCB1 = np.asarray(regrets_UCB1)
avg_regrets_Dec_UCB = regrets_Dec_UCB.mean(axis=0)
avg_regrets_UCB1 = regrets_UCB1.mean(axis=0)

# plot results - this section is highly arbitrary and should be edited to meet your needs
if numAgents > 10: 
    
    # plot worst Dec_UCB agent vs best UCB1 agent
    plt.figure(figsize=(5,5))
    plt.plot(range(args.time + 1), avg_regrets_Dec_UCB[np.argmax(avg_regrets_Dec_UCB[:, -1])])
    plt.plot(range(args.time + 1), avg_regrets_UCB1[np.argmin(avg_regrets_UCB1[:, -1])])
    plt.xlabel("Time")
    plt.ylabel("Expected Cumulative Regret")
    labels = ['Worst Decentralized Regret', 'Best UCB1 Regret']
    plt.legend(labels)
else: # plot all Dec_UCB agents against best UCB1 agent

    # display graph next to plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    ax = axes.flatten()
    for i in range(len(avg_regrets_Dec_UCB)):
        ax[0].plot(range(args.time + 1), avg_regrets_Dec_UCB[i])
    ax[0].plot(range(args.time + 1), avg_regrets_UCB1[np.argmin(avg_regrets_UCB1[:, -1])], '--')
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Expected Cumulative Regret")
    labels = ["Agent " + str(i) for i in range(numAgents)]
    labels.append('UCB1')
    ax[0].legend(labels)
    if G.number_of_nodes() <= 10:
        nx.draw_networkx(G, ax=ax[1], pos=nx.spring_layout(G))
        ax[1].set_axis_off()

    # just show plot, no graph
    # plt.figure(figsize=(5,5))
    # for i in range(len(avg_regrets_Dec_UCB)):
    #     plt.plot(range(args.time + 1), avg_regrets_Dec_UCB[i])
    # plt.plot(range(args.time + 1), avg_regrets_UCB1[np.argmin(avg_regrets_UCB1[:, -1])], '--')
    # plt.xlabel("Time")
    # plt.ylabel("Expected Cumulative Regret")
    # labels = ["Agent " + str(i) for i in range(numAgents)]
    # labels.append('UCB1')
    # plt.legend(labels)
plt.show()
    




