#!/usr/bin/env python
# coding: utf-8

# In[7]:


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

M = 6 # number of arms

#G = nx.DiGraph() # directed_rooted.png from this
#G.add_nodes_from([0, 1, 2, 3, 4, 5])
#G.add_edges_from([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
#                 (0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (5, 2)])

G = nx.fast_gnp_random_graph(6, 0.6, seed=1,directed=True) # directed_strongly_connected.png from this
print(nx.is_strongly_connected(G)) # check directed_strongly_connected.png with this

#G = nx.fast_gnp_random_graph(6, 0.5, seed=40,directed=False) # test.png from this
#G = nx.fast_gnp_random_graph(5, 0.25, seed=42,directed=False) #undirected_unconnected.png from this
#G.add_nodes_from([5])
#G = nx.complete_graph(6)

#G = nx.Graph() # graph for single agent
#G.add_nodes_from([0]) # for single agent

#G = nx.DiGraph()
#G.add_nodes_from([0, 1, 2, 3, 4, 5])
#G.add_edges_from([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
#                  (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])

nodes = list(G.nodes)
for i in nodes:
    G.add_edge(i,i)
A = nx.adjacency_matrix(G)

N = len(G) # number of agents
T = 250


# In[8]:


a = A.toarray() # make adjacency matrix an array for ease of use
neighbors = [] # list of all agents' neighbors
for i in range(len(a)):
    curr_neighbors = [] # neighbors of current agent
    for j in range(len(a)):
        if a[j][i] == 1:
            curr_neighbors.append(j)
    neighbors.append(curr_neighbors)

num_neighbors = [sum(A.toarray()[:,i]) for i in range(N)] # get cardinality of neighbors for each agent


# In[9]:


agent0_regret = [] # FIX THIS
agent1_regret = []
agent2_regret = []
agent3_regret = []
agent4_regret = []
agent5_regret = []

for epoch in range(1000):
    # initialize all vectors in a matrix for each time step, names corresponding to paper
    n = [np.zeros((N,M)) for t in range(T+1)]
    m = [np.zeros((N,M)) for t in range(T+1)]
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
            m[1][agent][arm] += 1
            z[1][agent][arm] = X[1][agent][arm]
            x[1][agent][arm] = X[1][agent][arm]
        
    for t in range(1,T): # loop over time
        for agent in range(N): # loop through all agents
            candidates = [] # candidate arms to choose
            Q = [] # corresponds to Q in paper

            for arm in range(M):
                if n[t][agent][arm] <= m[t][agent][arm] - M: # check decision making criteria
                    candidates.append(arm)
                else: 
                    q = z[t][agent][arm] + np.sqrt((4*np.log(t))/(3*num_neighbors[agent]*m[t][agent][arm]))
                    Q.append(q)

            if len(candidates) > 0: # check decision making criteria
                candidate = random.choice(candidates)
            else:
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
                    zsum += z[t][neighbor][arm] # calculate sum for z update
                    m[t+1][agent][arm] = max(n[t+1][agent][arm], m[t][neighbor][arm]) # update m considering all neighbors
                z[t+1][agent][arm] = (1/num_neighbors[agent])*(zsum + x[t+1][agent][arm] - x[t][agent][arm]) # update current agent's z
    rwds_tnspose = np.transpose(rwds) # transpose rwds to make it easier to plot
    for agent in range(len(rwds_tnspose)):
        regret = []
        for t in range(len(rwds_tnspose[agent])):
            avg = np.sum(rwds_tnspose[agent][0:t+1])/(t+1)
            regret.append(max_mean-avg)
    
        regret = np.cumsum(regret)
        if agent == 0: # FIX THIS
            agent0_regret.append(regret)
        elif agent == 1:
            agent1_regret.append(regret)
        elif agent == 2:
            agent2_regret.append(regret)
        elif agent == 3:
            agent3_regret.append(regret)
        elif agent == 4:
            agent4_regret.append(regret)
        else:
            agent5_regret.append(regret)


# In[10]:


# FIX THIS
arrays0 = [np.array(x) for x in agent0_regret]
avg_regret0 = [np.mean(k) for k in zip(*arrays0)]
arrays1 = [np.array(x) for x in agent1_regret]
avg_regret1 = [np.mean(k) for k in zip(*arrays1)]
arrays2 = [np.array(x) for x in agent2_regret]
avg_regret2 = [np.mean(k) for k in zip(*arrays2)]
arrays3 = [np.array(x) for x in agent3_regret]
avg_regret3 = [np.mean(k) for k in zip(*arrays3)]
arrays4 = [np.array(x) for x in agent4_regret]
avg_regret4 = [np.mean(k) for k in zip(*arrays4)]
arrays5 = [np.array(x) for x in agent5_regret]
avg_regret5 = [np.mean(k) for k in zip(*arrays5)]


# In[11]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax = axes.flatten()
# FIX THIS
ax[0].plot(range(T+1),avg_regret0)
ax[0].plot(range(T+1),avg_regret1)
ax[0].plot(range(T+1),avg_regret2)
ax[0].plot(range(T+1),avg_regret3)
ax[0].plot(range(T+1),avg_regret4)
ax[0].plot(range(T+1),avg_regret5)

ax[0].set_xlabel("Time")
ax[0].set_ylabel("Expected Cumulative Regret")
labels = ["Agent " + str(i) for i in range(N)]
ax[0].legend(labels)


nx.draw_networkx(G, ax=ax[1], pos=nx.spring_layout(G))
ax[1].set_axis_off()

#plt.savefig("directed_path.png")
plt.show()


# In[ ]:




