import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw


def calculate_weights(N, neighbors, num_neighbors, theorem):
    ''' Returns weight matrix W corresponding to Theorem 1 or Theorem 2, depending on theorem. '''

    W = np.zeros((N, N)) # W[i][j] = 0 if j not neighbor of i

    if theorem == 1 or N == 1:
            for agent in range(N):
                for neighbor in neighbors[agent]:
                    W[agent][neighbor] = 1 / num_neighbors[agent]
    else:
        for agent in range(N):
            for neighbor in neighbors[agent]:
                if agent != neighbor:
                    W[agent][neighbor] = 1 / max(num_neighbors[agent], num_neighbors[neighbor])
                else:
                    weight = 0
                    for nbor in neighbors[agent]:
                        if nbor != agent:
                            weight += (1 / max(num_neighbors[agent], num_neighbors[nbor]))
                    weight = 1 - weight
                    W[agent][neighbor] = weight
    return W

def generate_random_graph(size, graph_type, probability):
    if graph_type == 'undirected':
        G = nx.fast_gnp_random_graph(size, probability, directed=False)
        while not nx.is_connected(G):
            G = nx.fast_gnp_random_graph(size, probability, directed=False)
    else:
        G = nx.fast_gnp_random_graph(size, probability, directed=True)
        if graph_type == 'strong':
            while not nx.is_strongly_connected(G):
                G = nx.fast_gnp_random_graph(size, probability, directed=True)
        else:
            while (nx.is_strongly_connected(G) and not nx.is_weakly_connected(G)) or not nx.is_weakly_connected(G):
                G = nx.fast_gnp_random_graph(size, probability, directed=True)
    # add self-loops
    nodes = list(G.nodes)
    for i in nodes:
        G.add_edge(i,i) 
    return G

# use Lambert W function to solve (7) for t (which is L)
def find_sol(rho, N):
    w1 = lambertw((rho*np.log(rho)) / (864*N**2))
    t1 = 12*N*w1 / np.log(rho)
    
    w2 = lambertw((rho*np.log(rho)) / (864*N**2), -1)
    t2 = 12*N*w2 / np.log(rho)
    # print('t1: ' + str(t1) + '; t2: ' + str(t2)) # afaik t2 is always chosen as max
    return max(t1, t2)

#set parameters here
N = 400 # slows down a bunch after 200 or so
graph_type = 'undirected' # options: 'undirected', 'strong', 'weak'
probability = 0.4

theorem = 2 if graph_type == 'undirected' else 1
L = [0 for n in range(N+1)] # store values of L in here

# for every n, generate random graph and calculate L
for n in range(1, N+1):
    if n % 10 == 0: 
        print('starting iteration: ', n)
    G = generate_random_graph(n, graph_type, probability)

    # neighbor census
    A = nx.adjacency_matrix(G)
    a = A.toarray()
    neighbors = [] # list of all agents' neighbors
    for i in range(len(a)):
        curr_neighbors = [] # neighbors of current agent
        for j in range(len(a)):
            if a[j][i] == 1: # j is a neighbor of i if (j,i) is an edge
                curr_neighbors.append(j)
        neighbors.append(curr_neighbors)
    num_neighbors = [sum(A.toarray()[:,i]) for i in range(n)] # get cardinality of neighbors for each agent

    # get weight matrix W and rho2
    W = calculate_weights(n, neighbors, num_neighbors, theorem) 
    eigenvals, _ = np.linalg.eig(W)
    eigenvals = np.abs(eigenvals)
    eigenvals.sort()
    rho2 = eigenvals[-2] if len(eigenvals) > 1 else eigenvals[0]
    # print("rho_2 = ", rho2)

    # solve for L and store answer
    el = find_sol(rho2, n)
    # print("L = ", el)
    L[n] = el

# plot results
plt.figure(figsize=(8,5))
plt.plot(range(N+1), L)
# line of best fit
m, b = np.polyfit(range(N+1), L, 1)
plt.plot(range(N+1), m*(range(N+1))+b)
plt.xlabel('Graph Size (N)')
plt.ylabel('L')
plt.ylim(bottom=0, top=350000)
title = 'L vs. N : p=' + str(probability) + "; " + graph_type
plt.title(title)
fname = 'LvN-' + str(N) + '-' + str(int(probability*100)) + '-' + str(graph_type) + '.png' #
# plt.savefig(fname)
plt.show()





