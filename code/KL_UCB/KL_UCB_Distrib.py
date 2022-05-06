import numpy as np
import networkx as nx
rng = np.random.default_rng(1)

def KL(p, q): # compute Kullback-Leibler divergence (d in paper). check edge cases.
    if (p == 0 and q == 0) or (p == 1 and q == 1) or p == 0:
        return 0
    elif q == 0 or q == 1:
        return np.inf
    else:
        return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def dKL(p, q): # derivative of KL wrt q, p is constant
    result = (p-q)/(q*(q - 1.0))
    return result

def newton(N, z, k, t, Q, precision = 1e-3, max_iterations = 500, epsilon=1e-12):
    p = z # from paper
    q = p + 0.1 # initial guess?
    converged = False

    for n in range(max_iterations):
        f = KL(p, q) - Q/N # rearrange upper confidence bound eqn
        df = dKL(p, q) # derivative of f is just derivative of KL
        
        if abs(df) < epsilon: # check denominator is not too small
            break
        
        qnew = q - f / df
        if(abs(qnew - q) < precision): # check for early convergence
            converged = True
            break
        q = qnew

    return q

rwd_means = [.2, .3, .4, .5, .6]
T = 1000
N = 5 # agents
M = 5 # arms
n = np.ones((N,M,T))
m = np.ones((N,M,T))
z = np.zeros((N,M,T))
x = np.zeros((N,M,T))

G = nx.fast_gnp_random_graph(N, 0.5,directed=False) # undirected
while not nx.is_connected(G):
    G = nx.fast_gnp_random_graph(N, 0.5,directed=False)
nodes = list(G.nodes)
for i in nodes:
    G.add_edge(i,i)
A = nx.adjacency_matrix(G)
a = A.toarray() # make adjacency matrix an array for ease of use
neighbors = [] # list of all agents' neighbors
for i in range(len(a)):
    curr_neighbors = [] # neighbors of current agent
    for j in range(len(a)):
        if a[j][i] == 1:
            curr_neighbors.append(j)
    neighbors.append(curr_neighbors)
num_neighbors = [sum(A.toarray()[:,i]) for i in range(N)] # get cardinality of neighbors for each agent

def Q(t, sigma, Ni):
    return 3*(1+sigma)*(np.log(t) + 3*np.log(np.log(t)))/(2*Ni)

def w(i, j):
    if i == j:
        sumval = 0
        for neighbor in neighbors[i]:
            sumval += 1/(max(num_neighbors[i], num_neighbors[neighbor]))
        return 1 - sumval
    else:
        return 1/(max(num_neighbors[i], num_neighbors[j]))

for agent in range(N):
    for arm in range(M):
        val = rng.uniform(rwd_means[arm]-.1, rwd_means[arm]+.1)
        z[agent, arm, 0] = val
        x[agent, arm, 0] = val

for t in range(T-1):
    for agent in range(N):
        A = set()
        for k in range(M):
            if n[agent, k, t] <= m[agent, k, t] - M:
                A.add(k)
        if len(A) == 0:
            a = np.argmax([newton(n[agent, arm, t], z[agent, arm, t], arm, t, Q(t, 1, num_neighbors[agent])) for arm in range(M)])
        else:
            a = rng.choice(tuple(A))
        rwd = rng.uniform(rwd_means[a]-.1, rwd_means[a]+.1)
        for arm in range(M):
            if arm == a:
                n[agent, arm, t+1] = n[agent, arm, t] + 1
            else:
                n[agent, arm, t+1] = n[agent, arm, t]
            x[agent, arm, t+1] = (1/n[agent, arm, t+1])*(np.sum(x[agent, arm, :]) + rwd)
            z[agent, arm, t+1] = np.sum([w(agent, j) * z[j, arm, t] for j in neighbors[agent]]) + x[agent, arm, t+1] - x[agent, arm, t]
            m[agent, arm, t+1] = max(n[agent, arm, t+1], *[m[j, arm, t] for j in neighbors[agent]])

print(m)