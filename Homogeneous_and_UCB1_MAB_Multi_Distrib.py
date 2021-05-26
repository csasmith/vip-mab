#!/usr/bin/env python
# coding: utf-8

# # Change q depending on graph

# In[20]:


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import truncnorm
import time

M = 10 # number of arms

#G = nx.fast_gnp_random_graph(6, 0.6, seed=1,directed=True) # directed_strongly_connected.png from this
#print(nx.is_strongly_connected(G)) # check directed_strongly_connected.png with this

G = nx.fast_gnp_random_graph(6, 0.5, seed=40,directed=False) # undirected
print(nx.is_connected(G))

#G = nx.fast_gnp_random_graph(6, 0.5, seed=3,directed=True) # weakly connected
#print(nx.is_strongly_connected(G))
#print(nx.is_weakly_connected(G))

nodes = list(G.nodes)
for i in nodes:
    G.add_edge(i,i)
A = nx.adjacency_matrix(G)

N = len(G) # number of agents
T = 1000


# In[21]:


a = A.toarray() # make adjacency matrix an array for ease of use
neighbors = [] # list of all agents' neighbors
for i in range(len(a)):
    curr_neighbors = [] # neighbors of current agent
    for j in range(len(a)):
        if a[j][i] == 1:
            curr_neighbors.append(j)
    neighbors.append(curr_neighbors)

num_neighbors = [sum(A.toarray()[:,i]) for i in range(N)] # get cardinality of neighbors for each agent


# In[22]:


distribs = np.zeros((N,M))
for i in range(len(distribs)):
    for j in range(len(distribs[0])):
        r = random.random()
        if r < 0.33:
            distribs[i][j] = 1
        elif r < 0.66:
            distribs[i][j] = 2


# In[23]:


start = time.time()
E = 100
agent_regrets = np.zeros((N,E,T+1))

for epoch in range(E):
    # initialize all vectors in a matrix for each time step, names corresponding to paper
    n = [np.zeros((N,M)) for t in range(T+1)]
    m = [np.zeros((N,M)) for t in range(T+1)]
    x = [np.zeros((N,M)) for t in range(T+1)]
    X = [np.zeros((N,M)) for t in range(T+1)]
    z = [np.zeros((N,M)) for t in range(T+1)]
    x_sums = np.zeros((N,M))

    # create rwds array to hold all rewards each agent picks
    rwds = [np.zeros(N) for t in range(T+1)]

    arm_means = [random.uniform(0.05, 0.95) for x in range(0, M)] # means between 0.1 and 1 for all arms
    max_mean = max(arm_means) # get max mean

    sigma = 0.05 # standard deviation
    var = sigma**2
    # initialization step
    for agent in range(N):
        for arm in range(M):
            alpha = arm_means[arm]*(arm_means[arm]*(1-arm_means[arm])/var - 1)
            beta = (1-arm_means[arm])*(arm_means[arm]*(1-arm_means[arm])/var - 1)
            if distribs[agent][arm] == 1:
                X[1][agent][arm] = np.random.binomial(size=1, n=1, p= arm_means[arm])#np.random.beta(alpha, beta)
            elif distribs[agent][arm] == 2:
                X[1][agent][arm] = np.random.beta(alpha, beta)
            else:
                X[1][agent][arm] = truncnorm((0 - arm_means[arm]) / sigma, 
                                            (1 - arm_means[arm]) / sigma, 
                                            loc=arm_means[arm], scale=sigma).rvs()
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
                    q = z[t][agent][arm] + np.sqrt((3*np.log(t))/(num_neighbors[agent]*n[t][agent][arm]))#np.sqrt((4*np.log(t))/(3*n[t][agent][arm]))
                    Q.append(q)

            if len(candidates) > 0: # check decision making criteria
                candidate = random.choice(candidates)
            else:
                candidate = np.argmax(Q)
            
            alpha = arm_means[candidate]*(arm_means[candidate]*(1-arm_means[candidate])/var - 1)
            beta = (1-arm_means[candidate])*(arm_means[candidate]*(1-arm_means[candidate])/var - 1)
            if distribs[agent][candidate] == 1:
                X[t+1][agent][candidate] = np.random.binomial(size=1, n=1, p= arm_means[candidate])#np.random.beta(alpha, beta)
            elif distribs[agent][candidate] == 2:
                X[t+1][agent][candidate] = np.random.beta(alpha, beta)
            else:
                X[t+1][agent][candidate] = truncnorm((0 - arm_means[candidate]) / sigma, 
                                                     (1 - arm_means[candidate]) / sigma, 
                                                     loc=arm_means[candidate], scale=sigma).rvs()
            rwds[t+1][agent] = X[t+1][agent][candidate] # keep track of chosen reward

            for arm in range(M): # update all arm estimations for this agent
                if arm == candidate: # if chosen arm
                    n[t+1][agent][arm] = n[t][agent][arm] + 1
                    x_sums[agent][arm] = x_sums[agent][arm] + X[t+1][agent][arm]
                    x[t+1][agent][arm] = (1/n[t+1][agent][arm])*x_sums[agent][arm]
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
        agent_regrets[agent][epoch] = regret
    if epoch % 10 == 0:
        print("Epoch " + str(epoch) + " finished")
avg_regrets = agent_regrets.mean(axis=1)
end = time.time()
print(end-start)


# In[24]:


start = time.time()
E = 100
agent_regrets_2 = np.zeros((N,E,T+1))

for epoch in range(E):
    # initialize all vectors in a matrix for each time step, names corresponding to paper
    n = [np.zeros((N,M)) for t in range(T+1)]
    m = [np.zeros((N,M)) for t in range(T+1)]
    x = [np.zeros((N,M)) for t in range(T+1)]
    X = [np.zeros((N,M)) for t in range(T+1)]
    z = [np.zeros((N,M)) for t in range(T+1)]
    x_sums = np.zeros((N,M))

    # create rwds array to hold all rewards each agent picks
    rwds = [np.zeros(N) for t in range(T+1)]

    arm_means = [random.uniform(0.05, 0.95) for x in range(0, M)] # means between 0.1 and 1 for all arms
    max_mean = max(arm_means) # get max mean

    sigma = 0.05 # standard deviation
    var = sigma**2
    # initialization step
    for agent in range(N):
        for arm in range(M):
            alpha = arm_means[arm]*(arm_means[arm]*(1-arm_means[arm])/var - 1)
            beta = (1-arm_means[arm])*(arm_means[arm]*(1-arm_means[arm])/var - 1)
            if distribs[agent][arm] == 1:
                X[1][agent][arm] = np.random.binomial(size=1, n=1, p= arm_means[arm])#np.random.beta(alpha, beta)
            elif distribs[agent][arm] == 2:
                X[1][agent][arm] = np.random.beta(alpha, beta)
            else:
                X[1][agent][arm] = truncnorm((0 - arm_means[arm]) / sigma, 
                                            (1 - arm_means[arm]) / sigma, 
                                            loc=arm_means[arm], scale=sigma).rvs()
            n[1][agent][arm] += 1
            m[1][agent][arm] += 1
            z[1][agent][arm] = X[1][agent][arm]
            x[1][agent][arm] = X[1][agent][arm]
        
    for t in range(1,T): # loop over time
        for agent in range(N): # loop through all agents
            Q = [] # corresponds to Q in paper

            for arm in range(M):
                q = x[t][agent][arm] + np.sqrt((2*np.log(t))/(n[t][agent][arm]))
                Q.append(q)
    
            candidate = np.argmax(Q)

            alpha = arm_means[candidate]*(arm_means[candidate]*(1-arm_means[candidate])/var - 1)
            beta = (1-arm_means[candidate])*(arm_means[candidate]*(1-arm_means[candidate])/var - 1)
            if distribs[agent][candidate] == 1:
                X[t+1][agent][candidate] = np.random.binomial(size=1, n=1, p= arm_means[candidate])#np.random.beta(alpha, beta)
            elif distribs[agent][candidate] == 2:
                X[t+1][agent][candidate] = np.random.beta(alpha, beta)
            else:
                X[t+1][agent][candidate] = truncnorm((0 - arm_means[candidate]) / sigma, 
                                                     (1 - arm_means[candidate]) / sigma, 
                                                     loc=arm_means[candidate], scale=sigma).rvs()
            rwds[t+1][agent] = X[t+1][agent][candidate] # keep track of chosen reward

            for arm in range(M): # update all arm estimations for this agent
                if arm == candidate: # if chosen arm
                    n[t+1][agent][arm] = n[t][agent][arm] + 1
                    x_sums[agent][arm] = x_sums[agent][arm] + X[t+1][agent][arm]
                    x[t+1][agent][arm] = (1/n[t+1][agent][arm])*x_sums[agent][arm]
                else: # if not chosen arm
                    n[t+1][agent][arm] = n[t][agent][arm]
                    x[t+1][agent][arm] = x[t][agent][arm] # not mentioned in paper but seems necessary

    rwds_tnspose = np.transpose(rwds) # transpose rwds to make it easier to plot
    for agent in range(len(rwds_tnspose)):
        regret = []
        for t in range(len(rwds_tnspose[agent])):
            avg = np.sum(rwds_tnspose[agent][0:t+1])/(t+1)
            regret.append(max_mean-avg)
    
        regret = np.cumsum(regret)
        agent_regrets_2[agent][epoch] = regret
avg_regrets_2 = agent_regrets_2.mean(axis=1)
end = time.time()
print(end-start)


# In[19]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax = axes.flatten()
for i in range(len(avg_regrets)):
    ax[0].plot(range(T+1),avg_regrets[i])

ax[0].plot(range(T+1),avg_regrets_2[0],'--')

ax[0].set_xlabel("Time")
ax[0].set_ylabel("Expected Cumulative Regret")
labels = ["Agent " + str(i) for i in range(N)]
labels.append('UCB1')
ax[0].legend(labels)

nx.draw_networkx(G, ax=ax[1], pos=nx.spring_layout(G))
ax[1].set_axis_off()

#plt.savefig("undirected_multi_3_distrib.png")
plt.show()


# In[25]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax = axes.flatten()
ax[0].plot(range(T+1), avg_regrets[np.argmax(avg_regrets[:,-1])])
ax[0].plot(range(T+1), avg_regrets_2[np.argmin(avg_regrets_2[:,-1])])

nx.draw_networkx(G, ax=ax[1], pos=nx.spring_layout(G))
ax[1].set_axis_off()


# In[ ]:




