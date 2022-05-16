import numpy as np
import scipy.stats as sps
import networkx as nx
import math
rng = np.random.default_rng(1)
import matplotlib.pyplot as plt

class Dist_KL_UCB:
    ''' Representation of a multi-agent bandit problem and a method to run the decentralized KL_UCB algorithm on this problem

        Attributes
        ----------
        G: An undirected NetworkX graph instance representing the network over which agents communicate.
            It is assumed each node already has a self-loop.
        T: The number of time steps the Dist_KL_UCB algorithm will run for.
        arm_distributions: A NxM array of scipy.stats probability distributions bounded on [0,1].
            Distributions in the same column (pertaining to same arm) must share the same mean.
        means: A list of arm means. Extracted from arm_distributions
        M: Number of arms. Extracted from length of arm_distributions
        N: Number of agents
        sigma: An arbitrary positive hyperparameter, usually fixed at 0.01.
        regrets: A NxT numpy ndarray of expected agent regrets from the most recent algorithm run

        Notes
        -----
        By default we assume the most general case of heterogeneous reward distributions, as evidenced by
        the NxM shape of arm_distributions. While admittedly clunky, all one must do for the homogeneous reward
        case is to pass in an NxM arm_distributions array where each row is identical.
    '''

    def __init__(self, T, arm_distributions, G=None, sigma=0.01):
        ''' Construct a multi-agent bandit problem instance 
        
            Parameters
            ----------
            T: The number of time steps the Dist_KL_UCB algorithm will run for.
            arm_distributions: A NxM array of scipy.stats probability distributions bounded on [0,1].
                Distributions in the same column (pertaining to same arm) must share the same mean.
            G (optional): An undirected NetworkX graph instance representing the network over which agents communicate.
                It is assumed each node already has a self-loop. If no G is passed in, a randomly generated 
                graph of size len(arm_distributions) is used.
                The number of nodes must match the number of rows in arm_distributions
            sigma (optional): An arbitrary positive hyperparameter, usually fixed at 0.01.

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
        self.sigma = sigma
        self.regrets = None

    def KL(self, p, q):
            ''' Compute Kullback-Leibler divergence between Bernoulli distributions of parameters p and q. 
            
                Note that we can use this even when arm distributions are not Bernoulli. In this case,
                p and q are the means of their respective distributions.
            '''
            if (math.isclose(p, 0) and math.isclose(q, 0)) or (math.isclose(p, 1) and math.isclose(q, 1)) or (p >= 1 and q >= 1):
                return 0
            elif math.isclose(q, 0) or math.isclose(q, 1) or q <= 0 or q >= 1:
                return np.inf
            elif math.isclose(p, 0) or p <= 0:
                return np.inf
            elif p >= 1:
                return 0
            else:
                return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

    def dKL(self, p, q):
            ''' Compute the derivative of the Bernoulli Kullback-Leibler divergence between p and q, with respect to q. '''
            return (p-q)/(q*(q - 1.0))

    def Q(self, t, Ni):
        if (t > 1): return 3*(1+self.sigma)*(np.log(t) + 3*np.log(np.log(t)))/(2*Ni)
        return 0

    def newton(self, n, p, Q, precision=1e-3, max_iterations=20, epsilon=1e-6):
        ''' Calculate upper confidence bound via newton's method 
        
            WARNING: This function works in that it efficiently finds greatest approx zero to f in (0,1).
            However, KL-UCB technically specifies that the returned value of q should be such that 
            f(q) <= 0. The q's returned by this function seemingly always satisfy f(q)>=0.
            Enforcing f(q) <= 0 does not work (times out) because f(q) converges to 0 from the right.
            If this is unacceptable, maybe look into other root finding methods like bisection.

            Parameters
            ----------
            n: Number of times a certain agent i has picked a certain arm k at time t
            z: z estimate of agent i for arm k at time t
            Q: The value of the Q method for time t and agent i
            precision: Arbitrarily small convergence threshold
            max_iterations: Limit on number of iterations Newton's method should run
            epsilon: A miscellaneous arbitrarily small limit

            Return
            ------
            An upper confidence bound for the given parameters. Should technically be within [0,1], see warning.
        ''' 
        delta = 0.1
        q0 = min(p + delta, 1-epsilon) # initial guess
        q = q0
        converged = False

        for i in range(max_iterations):
            # if (p / q <= 0 or (1-p)/(1-q) <= 0): # sanity check for log domain errors
                # print(f'log error: p={p}, q={q}, i={i}')
            f = self.KL(p, q) - Q/n # rearrange upper confidence bound eqn
            df = self.dKL(p, q) # derivative of f is just derivative of KL
            if abs(df) < epsilon: break# check denominator is not too small
            qnew = max(q0, min(q - (f / df), 1-epsilon)) # chris: my approach for keeping q in (p,1)
            # print(f'q={q}, f(q)={f}, qnew={qnew}, precision={precision} n={n}')
            if(abs(qnew - q) < precision and abs(f) < precision): # check for early convergence
                converged = True
                # print(f'converged at {n} iterations')
                break
            q = qnew
        # if (not converged): print(f'did not converge')
        return q

    def plot_regret(self):
        ''' Plots regret of best and worst agent from last run vs theoretical regret bounds 

            Note: make sure Dist_KL_UCB.run() was called before calling this method
        '''
        optimal_arm = np.argmax(self.means)
        time_axis = list(range(self.T))
        # TODO: these are single agent theoretical regret bounds. Change to bounds in new paper
        coeff = 0
        for i in range(self.M):
            if (i != optimal_arm): coeff += (self.means[optimal_arm] - self.means[i]) / (self.KL(self.means[i], self.means[optimal_arm]))
        theoretical_regret_bounds = [coeff * np.log(t+1) for t in time_axis] # not sure if allowed to do this bc of lim sup, seems like it works tho
        plt.plot(time_axis, theoretical_regret_bounds, '--')
        plt.plot(time_axis, self.regrets[np.argmin(self.regrets[:, -1])])
        plt.plot(time_axis, self.regrets[np.argmax(self.regrets[:, -1])])
        plt.show()

    def run(self):
        ''' Run Dist_KL_UCB on the bandit problem held by self 
        
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
                    ucbs = [self.newton(n[agent, arm, t-1], z[agent, arm, t-1], self.Q(t-1, num_neighbors[agent])) for arm in range(self.M)]
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
# distkl = Dist_KL_UCB(T, distributions, sigma=0.01)
# distkl.run()
# distkl.plot_regret()

