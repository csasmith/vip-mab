#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

M = 5 # number of arms
G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3])
#G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
#G = nx.wheel_graph(N)
nodes = list(G.nodes)
for i in nodes:
    G.add_edge(i,i)
A = nx.adjacency_matrix(G)

N = len(G) # number of agents
T = 1000


# In[2]:


a = A.toarray() # make adjacency matrix an array for ease of use
neighbors = [] # list of all agents' neighbors
for i in range(len(a)):
    curr_neighbors = [] # neighbors of current agent
    for j in range(len(a)):
        if a[i][j] == 1:
            curr_neighbors.append(j)
    neighbors.append(curr_neighbors)

num_neighbors = [sum(A.toarray()[i]) for i in range(N)] # get cardinality of neighbors for each agent


# In[3]:


# initialize all vectors in a matrix for each time step, names corresponding to paper
n = [np.zeros((N,M)) for t in range(T)]
x = [np.zeros((N,M)) for t in range(T)]
X = [np.zeros((N,M)) for t in range(T)]
z = [np.zeros((N,M)) for t in range(T)]

# create rwds array to hold all rewards each agent picks
rwds = [np.zeros(N) for t in range(T)]

arm_means = [random.uniform(0, 1) for x in range(0, M)] # means between 0 and 1 for all arms
max_mean = max(arm_means) # get max mean

sigma = 0.1 # standard deviation


# In[4]:


# initialization step
for agent in range(N):
    for arm in range(M):
        X[0][agent][arm] = np.random.normal(arm_means[arm], sigma)
        n[0][agent][arm] += 1
        z[0][agent][arm] = X[0][agent][arm]
        x[0][agent][arm] = X[0][agent][arm]


# In[5]:


for t in range(0,T-1): # loop over time
    for agent in range(N): # loop through all agents
        Q = [] # corresponds to Q in paper
        
        for arm in range(M):
            q = z[t][agent][arm] + np.sqrt((4*np.log(t))/(num_neighbors[agent]*n[t][agent][arm]))
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


# In[6]:


n[T-1] # check counts at end


# In[7]:


arm_means


# In[8]:


rwds


# In[9]:


rwds_tnspose = np.transpose(rwds) # transpose rwds to make it easier to plot


# In[10]:


#max_rwds = np.ones(T)*max_mean
#cumu_max_rwds = np.cumsum(max_rwds)
#for agent in range(len(rwds_tnspose)):
#    cumu_actual_rwds = np.cumsum(rwds_tnspose[agent])
#    regret = (cumu_max_rwds - cumu_actual_rwds)
#    plt.scatter(range(T), regret)

#plt.xlabel("Time")
#plt.ylabel("Cumulative Regret")


# In[11]:


#max_rwds = np.ones(T)*max_mean
#cumu_max_rwds = np.cumsum(max_rwds)
#for agent in range(len(rwds_tnspose)):
#    regret = []
#    for t in range(len(rwds_tnspose[agent])):
#        avg = np.sum(rwds_tnspose[agent][0:t+1])/(t+1)
#        regret.append(max_mean-avg)
    
#    plt.scatter(range(T), regret)

plt.xlabel("Time")
plt.ylabel("Regret")


# In[12]:


max_rwds = np.ones(T)*max_mean
cumu_max_rwds = np.cumsum(max_rwds)
for agent in range(len(rwds_tnspose)):
    regret = []
    for t in range(len(rwds_tnspose[agent])):
        avg = np.sum(rwds_tnspose[agent][0:t+1])/(t+1)
        regret.append(max_mean-avg)
    
    regret = np.cumsum(regret)
    plt.scatter(range(T), regret)

plt.xlabel("Time")
plt.ylabel("Cumulative Regret")


# In[ ]:




