#!/usr/bin/env python
# coding: utf-8


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

M = 10 # number of arms
G = nx.Graph()
G = nx.fast_gnp_random_graph(10, 0.6, seed=1,directed=True)
print(nx.is_strongly_connected(G))

#G.add_nodes_from([0, 1, 2, 3])
#G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
#G = nx.fast_gnp_random_graph(6, 0.5, seed=40,directed=False) # test.png from this
#G = nx.fast_gnp_random_graph(6, 0.25, seed=42,directed=False) #undirected_unconnected.png from this
#G.add_nodes_from([6])
#G = nx.complete_graph(6)

nodes = list(G.nodes)
for i in nodes:
    G.add_edge(i,i)
A = nx.adjacency_matrix(G)

N = len(G) # number of agents
T = 2000

random.seed(42)



a = A.toarray() # make adjacency matrix an array for ease of use
neighbors = [] # list of all agents' neighbors
for i in range(len(a)):
    curr_neighbors = [] # neighbors of current agent
    for j in range(len(a)):
        if a[j][i] == 1:
            curr_neighbors.append(j)
    neighbors.append(curr_neighbors)

num_neighbors = [sum(A.toarray()[:,i]) for i in range(N)] # get cardinality of neighbors for each agent



# initialize all vectors in a matrix for each time step, names corresponding to paper
n = [np.zeros((N,M)) for t in range(T+1)]
x = [np.zeros((N,M)) for t in range(T+1)]
X = [np.zeros((N,M)) for t in range(T+1)]
z = [np.zeros((N,M)) for t in range(T+1)]

# create rwds array to hold all rewards each agent picks
rwds = [np.zeros(N) for t in range(T+1)]

arm_means = [random.uniform(0, 1) for x in range(0, M)] # means between 0 and 1 for all arms
max_mean = max(arm_means) # get max mean

sigma = 0.1 # standard deviation




# initialization step
for agent in range(N):
    for arm in range(M):
        X[1][agent][arm] = np.random.normal(arm_means[arm], sigma)
        n[1][agent][arm] += 1
        z[1][agent][arm] = X[0][agent][arm]
        x[1][agent][arm] = X[0][agent][arm]



for t in range(1,T): # loop over time
    for agent in range(N): # loop through all agents
        Q = [] # corresponds to Q in paper
        
        for arm in range(M):
            q = z[t][agent][arm] + np.sqrt((2*np.log(t))/(num_neighbors[agent]*n[t][agent][arm]))
            Q.append(q)

        candidate = np.argmax(Q)
        
        X[t+1][agent][candidate] = np.random.normal(arm_means[candidate], sigma) # calculate reward for current agent's arm
        rwds[t+1][agent] = X[t+1][agent][candidate] # keep track of chosen reward

        for arm in range(M): # update all arm estimations for this agent
            if arm == candidate: # if chosen arm
                n[t+1][agent][arm] = n[t][agent][arm] + 1
                xsum = 0
                for time in range(t+1): # sum up all rewards so far
                    xsum += X[time][agent][arm]
                x[t+1][agent][arm] = (1/n[t+1][agent][arm])*xsum
            else: # if not chosen arm
                n[t+1][agent][arm] = n[t][agent][arm]
                x[t+1][agent][arm] = x[t][agent][arm] # not mentioned in paper but seems necessary

            zsum = 0
            for neighbor in neighbors[agent]: # look at current agent's neighbors
                zsum += z[t][neighbor][arm] + x[t+1][agent][arm] - x[t][agent][arm] # calculate sum for z update
            z[t+1][agent][arm] = (1/num_neighbors[agent])*zsum # update current agent's z






rwds_tnspose = np.transpose(rwds) # transpose rwds to make it easier to plot


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax = axes.flatten()

for agent in range(len(rwds_tnspose)):
    regret = []
    for t in range(len(rwds_tnspose[agent])):
        avg = np.sum(rwds_tnspose[agent][0:t+1])/(t+1)
        regret.append(max_mean-avg)
    
    regret = np.cumsum(regret)
    ax[0].scatter(range(T+1), regret)


ax[0].set_xlabel("Time")
ax[0].set_ylabel("Cumulative Regret")
labels = ["Agent " + str(i) for i in range(N)]
ax[0].legend(labels)
ax[0].set_title("Random Graph")

nx.draw_networkx(G, ax=ax[1])
ax[1].set_axis_off()

#plt.savefig("undirected_unconnected.png")
plt.show()





