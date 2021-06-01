''' Exposes the Dec_UCB class for running the Dec_UCB algorithm with arbitrary parameters.'''

import networkx as nx
import numpy as np
import random

class Dec_UCB:
    ''' The Dec_UCB class allows us to run the Dec_UCB algorithm with arbitrary parameters.

        Create a Dec_UCB instance with the desired parameters and call run() on the instance.

        Attributes
        ----------
        G: A NetworkX graph instance representing the network over which agents communicate.
            It is assumed each node already has a self-loop.
        T: The number of time steps the Dec_UCB algorithm will run for.
        opcode: Either 1 or 2. The opcode tells us which upper confidence bound function and weights
            we should use. 1 corresponds to Theorem 1 and 2 corresponds to Theorem 2.
        arm_means: An array of floats bounded on [0,1] that represent the mean reward value for an arm.
            In other words, the list of mu_k's.
        distributions: A N x M (N agents, M arms) array of fixed scipy.stats probability distributions
            where each distribution corresponds to an agent-arm pair. We could sample from the 0th arm 
            for the 0th agent as follows: distributions[0][0].rvs()
        N: The number of agents.
        M: The number of arms.
        neighbors: An array containing the set of all neighbors for each agent
        num_neighbors: An array containing the size of the neighbor set for each agent.

        Methods
        -------
        run()
            Runs the Dec_UCB algorithm 
        '''

    def __init__(self, G, T, opcode, arm_means, distributions):
        ''' Construct a Dec_UCB instance.

        Parameters
        ----------
        G: A NetworkX graph instance representing the network over which agents communicate.
            It is assumed each node already has a self-loop.
        T: The number of time steps the Dec_UCB algorithm will run for.
        opcode: Either 1 or 2. The opcode tells us which upper confidence bound function and weights
            we should use. 1 corresponds to Theorem 1 and 2 corresponds to Theorem 2.
        arm_means: An array of floats bounded on [0,1] that represent the mean reward value for an arm.
            In other words, the list of mu_k's.
        distributions: A N x M (N agents, M arms) array of fixed scipy.stats probability distributions
            where each distribution corresponds to an agent-arm pair. We could sample from the 0th arm 
            for the 0th agent as follows: distributions[0][0].rvs()
        '''

        self.G = G # networkx graph
        self.T = T # number of time steps
        self.opcode = opcode # opcode is either 1 or 2
        self.arm_means = arm_means # size M array of mu_k's
        self.distributions = distributions # N x M distributions array

        self.N = G.number_of_nodes() # number of agents
        self.M = len(arm_means) # number of arms

        A = nx.adjacency_matrix(G)
        a = A.toarray()
        neighbors = [] # list of all agents' neighbors
        for i in range(len(a)):
            curr_neighbors = [] # neighbors of current agent
            for j in range(len(a)):
                if a[j][i] == 1: # j is a neighbor of i if (j,i) is an edge
                    curr_neighbors.append(j)
            neighbors.append(curr_neighbors)
        self.neighbors = neighbors
        self.num_neighbors = [sum(A.toarray()[:,i]) for i in range(self.N)] # get cardinality of neighbors for each agent

    def theorem1_ucb(self, t, n):
        ''' Upper confidence bound function corresponding to Theorem 1.'''

        return np.sqrt((4 * np.log(t)) / (3 * n))

    def theorem2_ucb(self, t, i, n):
        ''' Upper confidence bound function corresponding to Theorem 2.'''

        return np.sqrt((3 * np.log(t)) / ((self.num_neighbors[i] * n)))

    def calculate_weight(self, i, j):
        ''' Calculates weights used in updating the local z estimate.'''

        # Theorem 1 weight
        if self.opcode == 1:
            return 1 / self.num_neighbors[i]
        
        # Theorem 2 weight
        if i != j:
            return 1 / max(self.num_neighbors[i], self.num_neighbors[j])
        w = 0
        for neighbor in self.neighbors:
            w += 1 / max(self.num_neighbors[i], self.num_neighbors[j])
        return 1 - w



    def run(self):
        ''' Run the Dec_UCB algorithm. 

        Returns:
            agent_regrets: an N x T array of expected cumulative regret for each agent at every time step.
        '''
        
        # shorten variable names for convenience
        G = self.G
        T = self.T
        opcode = self.opcode
        arm_means = self.arm_means
        distributions = self.distributions
        N = self.N
        M = self.M
        neighbors = self.neighbors
        num_neighbors = self.num_neighbors

        # initialize data structures - names resemble variables in paper
        n = [np.zeros((N,M)) for t in range(T+1)] # local sample counters
        m = [np.zeros((N,M)) for t in range(T+1)] # local estimates of maximal global number of arm pulls
        x = [np.zeros((N,M)) for t in range(T+1)] # local sample means
        X = [np.zeros((N,M)) for t in range(T+1)] # realized rewards from arm pulls
        z = [np.zeros((N,M)) for t in range(T+1)] # local estimates of global arm sample means

        x_sums = np.zeros((N,M)) # used to calculate running sums of rewards
        rwds = [np.zeros(N) for t in range(T+1)] # create rwds array to hold all rewards each agent receives

        # initialization step
        for agent in range(N):
            for arm in range(M):
                X[1][agent][arm] = distributions[agent][arm].rvs()
                n[1][agent][arm] += 1
                m[1][agent][arm] += 1
                z[1][agent][arm] = X[1][agent][arm]
                x[1][agent][arm] = X[1][agent][arm]

        # main loop
        for t in range(1, T):
            for agent in range(N):
                
                # Choose arm
                candidates = [] # candidate arms to choose from
                Q = [] # corresponds to Q in paper
                for arm in range(M):
                    if n[t][agent][arm] <= m[t][agent][arm] - M: # check decision making criteria
                        candidates.append(arm)
                    else: 
                        if opcode == 1:
                            q = z[t][agent][arm] + self.theorem1_ucb(t, n[t][agent][arm])
                        else:
                            q = z[t][agent][arm] + self.theorem2_ucb(t, agent, n[t][agent][arm])
                        Q.append(q)
                if len(candidates) > 0: # check decision making criteria
                    candidate = random.choice(candidates)
                else:
                    candidate = np.argmax(Q)

                # Sample arm
                X[t+1][agent][candidate] = distributions[agent][arm].rvs()
                rwds[t+1][agent] = X[t+1][agent][candidate]

                # Update local variables
                for arm in range(M):
                    # update n and x
                    if arm == candidate:
                        n[t+1][agent][arm] = n[t][agent][arm] + 1
                        x_sums[agent][arm] += X[t+1][agent][arm]
                        x[t+1][agent][arm] = (1/n[t+1][agent][arm])*x_sums[agent][arm]
                    else:
                        n[t+1][agent][arm] = n[t][agent][arm]
                        x[t+1][agent][arm] = x[t][agent][arm]
                    # update z and m, these require looping over an agent's neighborhood
                    zsum = 0 # weighted summation portion of updated z value
                    for neighbor in neighbors[agent]:
                        w = self.calculate_weight(agent, neighbor)
                        zsum += w * z[t][neighbor][arm]
                        m[t+1][agent][arm] = max(n[t+1][agent][arm], m[t][neighbor][arm])
                    z[t+1][agent][arm] = (zsum + x[t+1][agent][arm] - x[t][agent][arm])
        
        # Algorithm finished. Use reward data to calculate and format our return value
        rwds_tnspose = np.transpose(rwds) # transpose rwds to make it easier to plot
        agent_regrets = []
        max_mean = max(arm_means)
        for agent in range(len(rwds_tnspose)):
            regret = []
            for t in range(len(rwds_tnspose[agent])):
                avg = np.sum(rwds_tnspose[agent][0:t+1]) / (t+1)
                regret.append(max_mean - avg)
            regret = np.cumsum(regret)
            agent_regrets.append(regret)
        return agent_regrets






                










