''' Run the Dec_UCB_Filter algorithm with arbitrary parameters.
    Dec_UCB_Filter is a modification of the Dec_UCB algorithm for problem instances with F Byzantine agents.
    At a high level, the modification involves agents trimming the F highest and F lowest incoming z_i,k(t)
    values at each t from their neighbors to obtain trimmed neighborhoods for each arm, then carrying out
    relevant Dec_UCB computations with these trimmed neighborhoods.
'''

# TODO: add byzantine behavior. Certain nodes should be labeled as Byzantine (read networkx docs),
# and can implement some byzantine behavior function that sets its variables appropriately

import networkx as nx
import numpy as np
import random


class Dec_UCB_Filter:
    ''' Create custom problem instances and run Dec_UCB_Filter on them

        Create a Dec_UCB_Filter instance with the desired parameters and call run() on the instance.

        Attributes
        ----------
        G: A NetworkX graph instance representing the network over which agents communicate.
            It is assumed each node already has a self-loop.
        T: The number of time steps the Dec_UCB_Filter algorithm will run for.
        arm_means: An array of floats bounded on [0,1] that represent the mean reward value for an arm.
            In other words, the list of mu_k's.
        distributions: A N x M (N agents, M arms) array of fixed scipy.stats probability distributions
            where each distribution corresponds to an agent-arm pair. We could sample from the 0th arm 
            for the 0th agent as follows: distributions[0][0].rvs()
        N: The number of agents.
        M: The number of arms.
        neighbors: An array containing the set of all neighbors for each agent
        num_neighbors: An array containing the size of the neighbor set for each agent.
        F: An upper bound on the number of Byzantine agents in the network

        Methods
        -------
        run()
            Runs the Dec_UCB_Filter algorithm 
        '''

    def __init__(self, G, T, arm_means, distributions, F):
        ''' Construct a Dec_UCB_Filter instance.

        Parameters
        ----------
        G: A NetworkX graph instance representing the network over which agents communicate.
            It is assumed each node already has a self-loop (caller responsible for this).
        T: The number of time steps the Dec_UCB_Filter algorithm will run for.
        arm_means: An array of floats bounded on (0,1) that represent the mean reward value for an arm.
            In other words, the list of mu_k's.
        distributions: A N x M (N agents, M arms) array of frozen scipy.stats probability distributions
            where each distribution corresponds to an agent-arm pair. We could sample from the 0th arm 
            for the 0th agent as follows: distributions[0][0].rvs()
        F: Upper bound on number of Byzantine agents in the network

        Raises
        ------
        TypeError
            If G is not a nonempty NetworkX Graph or DiGraph instance
        ValueError
            If T is not a positive integer, if any arm mean lies outside of (0,1), 
            if not every node in G has a self loop, if F < 0 or F >= N

        ''' 

        if G is None or not (isinstance(G, nx.classes.graph.Graph)
                or isinstance(G, nx.classes.digraph.DiGraph)):
            raise TypeError("G needs to be a nonempty NetworkX Graph or Digraph instance")
        if T < 1 or type(T) is not int:
            raise ValueError("T needs to be a positive integer")
        if any(mu <= 0 or mu >= 1 for mu in arm_means):
            raise ValueError("Arm means must lie in (0,1)")
        if nx.number_of_selfloops(G) != nx.number_of_nodes(G):
            raise ValueError("Every node should have a self-loop")
        if F < 0 or F >= G.number_of_nodes():
            raise ValueError("Number of Byzantine agents should be between 0 and N-1, inclusive")
        # problem instance parameters
        self.G = G # communication graph
        self.T = T # number of time steps
        self.arm_means = arm_means # size M array of mu_k's
        self.distributions = distributions # N x M distributions array
        self.N = G.number_of_nodes() # number of agents
        self.M = len(arm_means) # number of arms
        self.F = F
        # algorithm data structures 
        # NOTE: put these here so calc_ucb could access n. Felt messy not to include m,x,z. May be much more memory intensive than old way...
        self.n = np.zeros((T+1, self.N, self.M)) # local sample counters
        self.m = np.zeros((T+1, self.N, self.M)) # local estimates of maximal global number of arm pulls
        self.x = np.zeros((T+1, self.N, self.M)) # local sample means
        self.z = np.zeros((T+1, self.N, self.M)) # local estimates of global arm sample means

        A = nx.adjacency_matrix(G)
        a = A.toarray()
        neighbors = [] # list of all agents' neighbors
        for i in range(len(a)):
            curr_neighbors = [] # neighbors of current agent
            for j in range(len(a)):
                if a[j][i] == 1: # j is a neighbor of i if (j,i) is an edge
                    curr_neighbors.append(j)
            neighbors.append(curr_neighbors)
        self.neighbors = neighbors # TODO: think networkx already gives us way of accessing neighbors.
        self.num_neighbors = [sum(A.toarray()[:,i]) for i in range(self.N)] # get cardinality of neighbors for each agent

    def calc_filtered_neighborhoods(self, t):
        ''' Returns filtered neighborhoods for each agent and arm at time t

            Filtered neigborhoods is a dictionary of dictionaries. Keyed by agent then arm.

            Parameters
            ----------
            t: time step of values to be filtered. At time t call this with t-1.
        '''
        filtered_neighborhoods = {}
        for agent in range(self.N):
            filtered_neighborhoods[agent] = {}
            for arm in range(self.M):
                neighbors_sorted_by_z = sorted(self.neighbors[agent], key=lambda i : self.z[t][i][arm])
                loweri = 0
                while loweri < self.F and self.z[t][neighbors_sorted_by_z[loweri]][arm] < self.z[t][agent][arm]:
                    loweri += 1
                upperi = len(neighbors_sorted_by_z)-1
                while len(neighbors_sorted_by_z)-1 - upperi < self.F and self.z[t][neighbors_sorted_by_z[upperi]][arm] > self.z[t][agent][arm]:
                    upperi -= 1
                filtered_neighborhoods[agent][arm] = neighbors_sorted_by_z[loweri:upperi+1]
        return filtered_neighborhoods

    def calc_filtered_confidence_width(self, agent, arm, t, filtered_nbhd_size):
        ''' Calculate confidence width for an agent with filtered neighborhood

        Will use Theorem 1 UCB if G is a DiGraph, and Theorem 2 UCB otherwise

        Parameters
        ----------
        agent: index of agent we are calculating UCB for
        arm: index of arm we are calculating UCB for
        t: The current time step.
        filtered_nbhd_size: size of filtered neighborhood for agent and arm
        '''

        n = self.n[t][agent][arm] # number of times agent has pulled arm until time t; n_i,k(t)
        if type(self.G) == nx.DiGraph:
            return np.sqrt((4 * np.log(t)) / (3 * n))
        else:
            return np.sqrt((3 * np.log(t)) / ((filtered_nbhd_size * n)))
                
    def calc_filtered_weights(self, agent, arm, filtered_neighborhoods):
        ''' Calculate weights based on an arm and the filtered neighborhood of an agent
            (and the filtered neighborhoods of that agent's neighbors, in the case of 
            an undirected graph).
            Returns a dictionary with neighbor indices as keys and weights as values.
        '''

        weights = {}
        N_i = filtered_neighborhoods[agent][arm]
        if type(self.G) == nx.DiGraph:
            for neighbor in N_i:
                weights[neighbor] = 1 / len(N_i)
        else:
            sum = 0
            for neighbor in N_i:
                if neighbor != agent:
                    N_j = filtered_neighborhoods[neighbor][arm]
                    w = 1 / max(len(N_i), len(N_j))
                    sum += w
                    weights[neighbor] = w
            weights[agent] = 1 - sum
        return weights

    def run(self):
        ''' Run the Dec_UCB_Filter algorithm. 

        Raises
        ------
        ValueError
            If realized reward is outside the [0,1] boundary

        Returns
        -------
        agent_regrets: an N x T array of expected cumulative regret for each agent at every time step.
        '''

        x_sums = np.zeros((self.N, self.M)) # used to calculate running sums of rewards
        rwds = np.zeros((self.T+1, self.N)) # create rwds array to hold all rewards each agent receives

        # initialization step t=0
        for agent in range(self.N):
            for arm in range(self.M):
                reward = self.distributions[agent][arm].rvs()
                # print('received (init) reward ' + str(X[1][agent][arm]))
                self.n[0][agent][arm] += 1
                self.m[0][agent][arm] += 1
                self.z[0][agent][arm] = reward
                self.x[0][agent][arm] = reward
                # TODO: compute filtered neighborhoods 

        # main loop
        for t in range(1, self.T):
            filtered_neighborhoods = self.calc_filtered_neighborhoods(t-1)
            for agent in range(self.N):
                # if t < 10:
                #     print('Agent ' + str(agent) + ' (T=' + str(t) + ')\n--------------')
                #     print('n: ' + str(n[t][agent]))
                #     print('x: ' + str(x[t][agent]))
                #     print('m: ' + str(m[t][agent]))
                #     print('z: ' + str(z[t][agent]))
                
                # Choose arm
                candidates = [] # candidate arms to choose from in 
                candidate = -1 # chosen arm
                Q = [] # corresponds to Q in paper
                for arm in range(self.M):
                    if self.n[t][agent][arm] <= self.m[t][agent][arm] - self.M: # check decision making criteria
                        candidates.append(arm)
                    else: 
                        width = self.calc_filtered_confidence_width(agent, arm, t, len(filtered_neighborhoods[agent][arm]))
                        q = self.z[t][agent][arm] + width
                        Q.append(q)
                if len(candidates) > 0: # check decision making criteria
                    candidate = random.choice(candidates)
                    # print('randomly chose candidate ' + str(candidate))
                else:
                    # print('Q ' + str(Q)) if t < 10 else None
                    candidate = np.argmax(Q)

                # print('Chose arm ' + str(candidate)) if t < 10 or T - t < 10 else None

                # Sample arm
                reward = self.distributions[agent][candidate].rvs()
                if reward < 0 or reward > 1:
                    raise ValueError("Rewards should be bounded on [0,1]")
                rwds[t+1][agent] = reward
                # print('received reward ' + str(rwds[t+1][agent]))

                # Update local variables
                for arm in range(self.M):
                    # update n and x
                    if arm == candidate:
                        self.n[t+1][agent][arm] = self.n[t][agent][arm] + 1
                        self.x[t+1][agent][arm] = ((self.x[t][agent][arm] * self.n[t][agent][arm]) + reward) / self.n[t+1][agent][arm]
                    else:
                        self.n[t+1][agent][arm] = self.n[t][agent][arm]
                        self.x[t+1][agent][arm] = self.x[t][agent][arm]
                    # update z and m, these require looping over an agent's filtered neighborhood
                    zsum = 0 # weighted summation portion of updated z value
                    m_vals = [self.n[t+1][agent][arm]] # for updating m_i,k(t+1). n_i,k(t+1) is in there for max comparison
                    weights = self.calc_filtered_weights(agent, arm, filtered_neighborhoods)
                    for neighbor in filtered_neighborhoods[agent][arm]:
                        w = weights[neighbor]
                        zsum += (w * self.z[t][neighbor][arm])
                        m_vals.append(self.m[t][neighbor][arm])
                    self.m[t+1][agent][arm] = max(m_vals)
                    self.z[t+1][agent][arm] = (zsum + self.x[t+1][agent][arm] - self.x[t][agent][arm])

                # print('--------------\n') if t < 10 else None
        
        # Algorithm finished. Use reward data to calculate and format our return value
        rwds_tnspose = np.transpose(rwds) # transpose rwds to make it easier to plot
        agent_regrets = []
        max_mean = max(self.arm_means)
        for agent in range(len(rwds_tnspose)):
            regret = []
            for t in range(len(rwds_tnspose[agent])):
                avg = np.sum(rwds_tnspose[agent][0:t+1]) / (t+1)
                regret.append(max_mean - avg)
            regret = np.cumsum(regret)
            agent_regrets.append(regret)
        return agent_regrets


# test it out







                










