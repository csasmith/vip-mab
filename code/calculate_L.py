import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw

def calculate_weights(N, neighbors, num_neighbors):
    ''' Returns weight matrix W corresponding to Theorem 1 or Theorem 2, depending on opcode. '''

    W = np.zeros((N, N)) # W[i][j] = 0 if j not neighbor of i
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

N = 100
G = nx.complete_graph(N)

# add self-loops
nodes = list(G.nodes)
for i in nodes:
    G.add_edge(i,i) 
A = nx.adjacency_matrix(G)
a = A.toarray()
neighbors = [] # list of all agents' neighbors
for i in range(len(a)):
    curr_neighbors = [] # neighbors of current agent
    for j in range(len(a)):
        if a[j][i] == 1: # j is a neighbor of i if (j,i) is an edge
            curr_neighbors.append(j)
    neighbors.append(curr_neighbors)
num_neighbors = [sum(A.toarray()[:,i]) for i in range(N)] # get cardinality of neighbors for each agent

W = calculate_weights(N,neighbors,num_neighbors) 
eigenvals, _ = np.linalg.eig(W)
eigenvals = np.abs(eigenvals)
eigenvals.sort()
rho2 = eigenvals[-2] if len(eigenvals) > 1 else eigenvals[0]

def find_sol(rho, N):
    w1 = lambertw((rho*np.log(rho)) / (864*N**2))
    t1 = 12*N*w1 / np.log(rho)
    
    w2 = lambertw((rho*np.log(rho)) / (864*N**2), -1)
    t2 = 12*N*w2 / np.log(rho)
    return max(t1, t2)

print("rho_2 = ", rho2)
print("N = ", N)
print("L = ", find_sol(rho2,N))