''' Perform desired Dec_UCB simulations from standard input '''

from Dec_UCB import Dec_UCB
import argparse
import networkx as nx
import numpy as np
import scipy.stats as sps
import random
import matplotlib.pyplot as plt

# parse arguments from standard input
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('-N', '--numAgents', type=int, default=5, help="Number of agents")
group.add_argument('-f', '--inputFile', help="text file path containing a NetworkX Graph " + 
        "or DiGraph in multiline adjacency list format. The graph must be nonempty and each node must " + 
        "have a self-loop. Additionally, the graph must match the type parameter")
parser.add_argument('type', choices=['strong', 'weak', 'undirected'], help="Graph type must " + 
        "be either strongly connected, weakly connected, or undirected connected")
# changed M to numArms to match with below usage
parser.add_argument('numArms', type=int, default='6', help="Number of arms")
parser.add_argument('setting', choices=['homogeneous', 'heterogeneous'], help="Arm distributions can " + 
        "be homogeneous or heterogeneous")
parser.add_argument('-m', '--means', type=float, nargs='+', help="List of M arm means within (0,1)")
parser.add_argument('-d', '--distributions', nargs='+', default=['truncnorm'], help="List of scipy " + 
        "probability distribution names")
parser.add_argument('-s', '--stddev', type=float, default=0.05, help="Standard deviation, " + 
        "if applicable to any distribution")
parser.add_argument('-t', '--time', type=int, default=1000, help="Number of time steps")
parser.add_argument('-e', '--epochs', type=int, default=100, help="Number of iterations " +  
        "that Dec_UCB is repeated for")
args = parser.parse_args()
print(str(args))

# if no means provided, have to generate defaults here once numArms is known
if args.means == None:
    args.means=[random.uniform(0.05, 0.95) for x in range(0, args.numArms)]

# additional validation - any validation that appears to be missing is most likely in Dec_UCB.py
if args.inputFile:
    G = nx.read_multiline_adjlist(args.inputFile)
    if args.type == 'strong' and not nx.is_strongly_connected(G):
        raise TypeError("Graph type must match type argument")
    if args.type == 'weak' and not nx.is_weakly_connected(G):
        raise TypeError("Graph type must match type argument")
    if args.type == 'undirected' and not nx.is_connected(G):
        raise TypeError("Graph type must match type argument")
if args.numAgents <= 0:
    raise ValueError("numAgents needs to be a positive integer")
if args.numArms <= 0:
    raise ValueError("numArms needs to be a positive integer")
if len(args.means) != args.numArms:
    raise ValueError("means needs to be a list of numArms floats between 0 and 1")
if args.stddev < 0:
    raise ValueError("standard deviation needs to be a non-negative float")
if args.time <= 0:
    raise ValueError('The number of time steps must be a positive integer')
if args.epochs <= 0:
    raise ValueError("The number of epochs must be a positive integer")
supported_distributions = {'truncnorm', 'bernoulli', 'beta', 'uniform'}
if any(d not in supported_distributions for d in args.distributions):
    raise ValueError("distributions must belong to the currently supported distributions. " + 
        "Supported distributions: " + str(supported_distributions))

# randomly generate graph if -f option not used
if args.numAgents:
    if args.type == 'strong':
        G = nx.fast_gnp_random_graph(args.numAgents, 0.5, directed=True)
        while not nx.is_strongly_connected(G):
            G = nx.fast_gnp_random_graph(args.numAgents, 0.5, directed=True)
    if args.type == 'weak':
        G = nx.fast_gnp_random_graph(args.numAgents, 0.5, directed=True)
        while not nx.is_weakly_connected(G):
            G = nx.fast_gnp_random_graph(args.numAgents, 0.5, directed=True)
    if args.type == 'undirected':
        G = nx.fast_gnp_random_graph(args.numAgents, 0.5, directed=False)
        while not nx.is_connected(G):
            G = nx.fast_gnp_random_graph(args.numAgents, 0.5, directed=False)
    # add self-loops
    nodes = list(G.nodes)
    for i in nodes:
        G.add_edge(i,i)

# get opcode from graph type
if args.type == 'strong' or args.type == 'weak':
    opcode = 1
else:
    opcode = 2

def set_distribution(d, j):
    
    # print('args.means[j] ' + str(args.means[j]) + ', and j is ' + str(j))
    if d == 'truncnorm':
        a = (0 - args.means[j]) / args.stddev
        b = (1 - args.means[j]) / args.stddev
        return sps.truncnorm(a, b, loc=args.means[j], scale=args.stddev)
    if d == 'bernoulli':
        return sps.bernoulli(args.means[j])
    if d == 'beta':
        alpha = args.means[j] * (args.means[j] * (1 - args.means[j]) / args.stddev**2 - 1)
        beta = (1 - args.means[j]) * (args.means[j] * (1 - args.means[j]) / args.stddev**2 - 1)
        return sps.beta(alpha, beta)
    if d == 'uniform':
        # we wish to obtain a uniform distribution given a certain mean
        # to do this, we pick the widest uniform distribution possible still in [0,1]
        radius = min(args.means[j], abs(1 - args.means[j]))
        return sps.uniform(loc=args.means[j] - radius, scale=2*radius) # should be [loc, loc + scale]

# create distributions array
distributions = [[None for i in range(args.numArms)] for i in range(args.numAgents)]
if args.setting == 'heterogeneous':
    for i in range(args.numAgents):
        for j in range(args.numArms):
            d = random.choice(args.distributions)
            distributions[i][j] = set_distribution(d, j)
else:
    for j in range(args.numArms):
        d = random.choice(args.distributions)
        # print('randomly chosen distribution is ' + str(d))
        for i in range(args.numAgents):
            distributions[i][j] = set_distribution(d, j)


print('args.means ' + str(args.means))
print('max_mean ' + str(max(args.means)))
print('distributions ' + str([[d.mean() for d in arr] for arr in distributions]))

# run simulations
if nx.number_of_nodes(G) <= 10: # if small graph, use same graph for all epochs
    regrets = []
    simulator = Dec_UCB(G, args.time, opcode, args.means, distributions)
    for e in range(args.epochs):
        regrets.append(simulator.run())
        print('epoch: ' + str(e)) if e % 10 == 0 else None
    regrets = np.asarray(regrets) # shape (E, N, T+1)
    avg_regrets = regrets.mean(axis=0) # shape (N, T+1)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    ax = axes.flatten()

    for i in range(len(avg_regrets)):
        ax[0].plot(range(args.time+1),avg_regrets[i])
        # print(str(avg_regrets[i]))

    #ax[0].plot(range(args.time+1),avg_regrets_2[0],'--')

    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Expected Cumulative Regret")
    labels = ["Agent " + str(i) for i in range(args.numAgents)]
    #labels.append('UCB1')
    ax[0].legend(labels)

    nx.draw_networkx(G, ax=ax[1], pos=nx.spring_layout(G))
    ax[1].set_axis_off()

    #plt.savefig("weakly_multi_3_distrib.eps", bbox_inches='tight')
    plt.show()
else:
    print('large, use different graph for each epoch... unless specific graph was given')
    # if file was given, we would probably want to use that for all simulations





