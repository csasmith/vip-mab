import networkx as nx
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

# environment parameters
T = 2000 # time steps
N = 4 # number agents
F = 1 # number of byzantine agents
distribution = sps.bernoulli(0.5) # use as distribution.rvs() 
G = nx.complete_graph(N)
for i in range(F):
    G.nodes[i]['byzantine'] = True

# data structures
z = np.zeros((T+1,N)) # z values
x = np.zeros((T+1,N)) # local mean estimates (these are shared, filtered, etc)

# byzantine strategy - for right now just transmit 1/3 
def byzantine_transmit_value():
    return 1/3

# t=0 initialization step
for agent, is_byzantine in G.nodes.data('byzantine'):
    x[0][agent] = distribution.rvs()
    z[0][agent] = distribution.rvs()
    if (is_byzantine):
        x[0][agent] = byzantine_transmit_value()
        
# main loop
for t in range(1, T+1):
    x_sorted = sorted(x[t-1,:])
    for agent, is_byzantine in G.nodes.data('byzantine'):
        if (not is_byzantine):
            # filter
            my_x_value = x[t-1][agent]
            loweri = 0
            while loweri < F and x_sorted[loweri] < my_x_value:
                loweri += 1
            upperi = len(x_sorted) - 1
            while len(x_sorted) - 1 - upperi < F and x_sorted[upperi] > my_x_value:
                upperi -= 1
            x_filtered = x_sorted[loweri:upperi+1] # NOTE: because complete graph, no need to preserve index of filtered x value
            # update z
            rwd = distribution.rvs()
            x[t][agent] = ((my_x_value * (t-1)) + rwd) / t
            weighted_sum = np.average(x_filtered)
            z[t][agent] = weighted_sum + x[t][agent] - my_x_value
        else:
            x[t][agent] = byzantine_transmit_value()

print(x[T-20:,:])
print(z[T-20:,:])

# plotting
time_axis = list(range(T+1))
for agent, is_byzantine in G.nodes.data('byzantine'):
    if (not is_byzantine):
        plt.plot(time_axis, z[:,agent])
plt.show()

