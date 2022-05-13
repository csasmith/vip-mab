import numpy as np
import scipy.stats as sps
import networkx as nx
rng = np.random.default_rng(1)
import matplotlib.pyplot as plt

class Dist_UCB1:
    ''' Representation of a multi-agent bandit problem and a method to run the decentralized UCB1 algorithm on this problem

        Attributes
        ----------
        G: An undirected NetworkX graph instance representing the network over which agents communicate.
            It is assumed each node already has a self-loop.
        T: The number of time steps the Dist_UCB1 algorithm will run for.
        arm_distributions: A NxM array of scipy.stats probability distributions bounded on [0,1].
            Distributions in the same column (pertaining to same arm) must share the same mean.
        means: A list of arm means. Extracted from arm_distributions
        M: Number of arms. Extracted from length of arm_distributions
        N: Number of agents
        beta: An arbitrary positive hyperparameter, usually fixed at 0.01.
        regrets: A NxT numpy ndarray of expected agent regrets from the most recent algorithm run

        Notes
        -----
        By default we assume the most general case of heterogeneous reward distributions, as evidenced by
        the NxM shape of arm_distributions. While admittedly clunky, all one must do for the homogeneous reward
        case is to pass in an NxM arm_distributions array where each row is identical.
    '''

    def __init__(self, T, arm_distributions, G=None, beta=0.01):
        ''' Construct a multi-agent bandit problem instance 
        
            Parameters
            ----------
            T: The number of time steps the Dist_UCB1 algorithm will run for.
            arm_distributions: A NxM array of scipy.stats probability distributions bounded on [0,1].
                Distributions in the same column (pertaining to same arm) must share the same mean.
            G (optional): An undirected NetworkX graph instance representing the network over which agents communicate.
                It is assumed each node already has a self-loop. If no G is passed in, a randomly generated 
                graph of size len(arm_distributions) is used.
                The number of nodes must match the number of rows in arm_distributions
            beta (optional): An arbitrary positive hyperparameter, usually fixed at 0.01.

            Raises
            ------
            TypeError
                If G is not an undirected NetworkX Graph with self-loops.
            ValueError
                If G is provided and the number of nodes does not match len(arm_distributions)
                If T is not a positive integer.
                If the support for any arm distribution is not in [0,1].
                If any two distributions in the same column do not share the same mean
        '''
        if (G is None):
            G = nx.fast_gnp_random_graph(len(arm_distributions), 0.5, directed=False)
            while not nx.is_connected(G):
                G = nx.fast_gnp_random_graph(len(arm_distributions), 0.5, directed=False)
            nodes = list(G.nodes)
            for i in nodes:
                G.add_edge(i,i) 
        if (not isinstance(G, nx.classes.graph.Graph)):
            raise TypeError("G needs to be an undirected NetworkX Gra instance")
        if nx.number_of_selfloops(G) != nx.number_of_nodes(G):
            raise ValueError("Every node should have a self-loop")
        if (G.number_of_nodes() != len(arm_distributions)):
            raise ValueError('The number of nodes must match the number of rows in arm_distributions')
        if T < 1 or type(T) is not int:
            raise ValueError("T needs to be a positive integer")
        arm_distributions = np.asarray(arm_distributions) # cast to numpy ndarray just in case it wasn't already
        for row in arm_distributions:
            if (any(d.support()[0] < 0 or d.support()[1] > 1 for d in row)): 
                raise ValueError('distribution support must lie in [0,1]')
        for col in arm_distributions.T:
            if (any(d.mean() != col[0].mean() for d in col)):
                raise ValueError('distribution means must be the same within a column')
        self.G = G
        self.N = G.number_of_nodes()
        self.T = T
        self.arm_distributions = arm_distributions
        self.means = [d.mean() for d in arm_distributions[0]]
        self.M = len(arm_distributions[0])
        self.beta = beta
        self.regrets = None

    def C(self, t, beta, n, Ni):
        return (1 + beta)*np.sqrt((3*np.log(t))/(Ni*n))

    def plot_regret(self):
        ''' Plots regret of best and worst agent from last run vs theoretical regret bounds 

            Note: make sure Dist_UCB1.run() was called before calling this method
        '''
        #optimal_arm = np.argmax(self.means)
        time_axis = list(range(self.T))
        # TODO: these are single agent theoretical regret bounds. Change to bounds in new paper
        #coeff = 0
        #for i in range(self.M):
        #    if (i != optimal_arm): coeff += (self.means[optimal_arm] - self.means[i]) / (self.KL(self.means[i], self.means[optimal_arm]))
        #theoretical_regret_bounds = [coeff * np.log(t+1) for t in time_axis] # not sure if allowed to do this bc of lim sup, seems like it works tho
        #plt.plot(time_axis, theoretical_regret_bounds, '--')
        plt.plot(time_axis, self.regrets[np.argmin(self.regrets[:, -1])])
        plt.plot(time_axis, self.regrets[np.argmax(self.regrets[:, -1])])
        plt.show()

    def run(self):
        ''' Run Dist_UCB1 on the bandit problem held by self 
        
            Return
            ------
            A NxT numpy ndarray with expected regrets of each agent at each time t
        '''
        # populate neighbors and num_neighbors lists
        A = nx.adjacency_matrix(self.G)
        a = A.toarray()
        neighbors = [] # list of all agents' neighbors
        for i in range(len(a)):
            curr_neighbors = [] # neighbors of current agent
            for j in range(len(a)):
                if a[j][i] == 1:
                    curr_neighbors.append(j)
            neighbors.append(curr_neighbors)
        num_neighbors = [sum(A.toarray()[:,i]) for i in range(self.N)] # get cardinality of neighbors for each agent
        # populate a NxN weights matrix
        W = np.zeros((self.N, self.N))
        for agent in range(self.N):
            for neighbor in neighbors[agent]:
                if (agent != neighbor):
                    W[agent][neighbor] = 1 / max(num_neighbors[agent], num_neighbors[neighbor])
                else:
                    s = 0
                    for nbor in neighbors[agent]:
                        if (nbor != agent): s += 1 / max(num_neighbors[agent], num_neighbors[nbor])
                    W[agent][neighbor] = 1 - s
        # our data structures
        n = np.ones((self.N, self.M, self.T))
        m = np.ones((self.N, self.M, self.T))
        z = np.zeros((self.N, self.M, self.T))
        x = np.zeros((self.N, self.M, self.T))
        exp_cum_rwds = np.zeros((self.N, self.T))
        # t=0 initialization
        for agent in range(self.N):
            for arm in range(self.M):
                val = self.arm_distributions[agent][arm].rvs()
                z[agent, arm, 0] = val
                x[agent, arm, 0] = val
        # main loop
        for t in range(1, self.T):
            for agent in range(self.N):
                A = set()
                for k in range(self.M):
                    if n[agent, k, t-1] <= m[agent, k, t-1] - self.M:
                        A.add(k)
                if len(A) == 0:
                    ucbs = [z[agent, arm, t-1] + self.C(t-1, self.beta, n[agent, arm, t-1], num_neighbors[agent]) for arm in range(self.M)]
                    a = np.argmax(ucbs)
                else:
                    a = rng.choice(tuple(A))
                rwd = self.arm_distributions[agent][a].rvs()
                # print(f'rwd={rwd}')
                exp_cum_rwds[agent][t] = exp_cum_rwds[agent][t-1] + self.means[a]
                # updates
                for arm in range(self.M):
                    if arm == a:
                        n[agent, arm, t] = n[agent, arm, t-1] + 1
                    else:
                        n[agent, arm, t] = n[agent, arm, t-1]
                    x[agent, arm, t] = ((n[agent, arm, t-1] * x[agent, arm, t-1]) + rwd * (arm == a)) / n[agent, arm , t]
                    z[agent, arm, t] = np.sum([W[agent][j] * z[j, arm, t-1] for j in neighbors[agent]]) + x[agent, arm, t] - x[agent, arm, t-1]
                    m[agent, arm, t] = max(n[agent, arm, t], *[m[j, arm, t-1] for j in neighbors[agent]])
        # compute regrets
        optimal_arm = np.argmax(self.means)
        optimal_exp_cum_rwds = [[t * self.means[optimal_arm] for t in range(self.T)] for n in range(self.N)]
        regrets = np.asarray(optimal_exp_cum_rwds) - exp_cum_rwds
        self.regrets = regrets
        return regrets

# # test run
# T = 1000
# N = 10
# rwd_means = [.2, .3, .4, .5, .6]
# distributions = [[sps.uniform(loc=rwd_means[i] - .1, scale=0.2) for i in range(len(rwd_means))] for n in range(N)]
# distucb1 = Dist_UCB1(T, distributions, beta=0.01)
# distucb1.run()
# distucb1.plot_regret()

