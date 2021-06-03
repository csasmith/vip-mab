''' Runs the UCB1 algorithm for comparison with Dec_UCB  '''

import numpy as np
import random

class UCB1:
    ''' Create a custom problem instance and run UCB1 on the instance

    The problem instances are of the same form as problem instances for Dec_UCB, except
    there is no communication and therefore no graph to communicate over, only a specified
    number of agents that each run UCB1 independently.

    Attributes
    ----------
    T: The number of time steps the UCB1 algorithm will run for.
    arm_means: An array of floats bounded on [0,1] that represent the mean reward value for an arm.
        In other words, the list of mu_k's.
    distributions: A N x M (N agents, M arms) array of fixed scipy.stats probability distributions
        where each distribution corresponds to an agent-arm pair. We could sample from the 0th arm 
        for the 0th agent as follows: distributions[0][0].rvs()
    N: The number of agents.
    M: The number of arms.

    Methods
    -------
    run()
        Runs the UCB1 algorithm 
    '''

    def __init__(self, T, arm_means, distributions, N):
        ''' Construct a Dec_UCB instance.

        Parameters
        ----------
        T: The number of time steps the UCB1 algorithm will run for.
        arm_means: An array of floats bounded on (0,1) that represent the mean reward value for an arm.
            In other words, the list of mu_k's.
        distributions: A N x M (N agents, M arms) array of frozen scipy.stats probability distributions
            where each distribution corresponds to an agent-arm pair. We could sample from the 0th arm 
            for the 0th agent as follows: distributions[0][0].rvs()
        N: The number of agents

        Raises
        ------
        ValueError
            If T is not a positive integer, if any arm mean lies outside of (0,1), 
            or if N is not a positive integer
        '''

        if T < 1 or type(T) is not int:
            raise ValueError("T needs to be a positive integer")
        
        if any(mu <= 0 or mu >= 1 for mu in arm_means):
            raise ValueError("Arm means must lie in (0,1)")

        if N <= 0 or type(N) is not int:
            raise ValueError("N must be a positive integer")

        self.T = T
        self.arm_means = arm_means
        self.distributions = distributions
        self.N = N
        self.M = len(arm_means)

    def ucb(self, t, n):
        ''' UCB1 upper confidence bound '''

        return np.sqrt((2*np.log(t))/n)

    def run(self):
        ''' Run the UCB1 algorithm
        
        Raises
        ------
        ValueError
            If realized reward is outside the [0,1] boundary

        Returns
        -------
        agent_regrets: an N x T array of expected cumulative regret for each agent at every time step.
        '''

        # shorter variable names for convenience
        N = self.N
        M = self.M
        T = self.T
        arm_means = self.arm_means
        distributions = self.distributions

        # initialize data structures
        n = [np.zeros((N, M)) for t in range(T+1)]
        x = [np.zeros((N, M)) for t in range(T+1)]
        rwds = [np.zeros(N) for t in range(T+1)]

        # initialization step
        for agent in range(N):
            for arm in range(M):
                reward = distributions[agent][arm].rvs()
                n[1][agent][arm] += 1
                x[1][agent][arm] = reward
        
        # main loop
        for t in range(1, T):
            for agent in range(N):
                # Choose arm
                Q = []
                for arm in range(M):
                    q = x[t][agent][arm] + self.ucb(t, n[t][agent][arm])
                    Q.append(q)
                candidate = np.argmax(Q)

                # Sample arm
                reward = distributions[agent][candidate].rvs()
                if reward < 0 or reward > 1:
                    raise ValueError('Rewards should be bounded on [0,1]')
                rwds[t+1][agent] = reward

                # Update variables
                for arm in range(M):
                    if arm == candidate:
                        n[t+1][agent][arm] = n[t][agent][arm] + 1
                        x[t+1][agent][arm] = ((x[t][agent][arm] * n[t][agent][arm]) + reward) / n[t+1][agent][arm]
                    else:
                        n[t+1][agent][arm] = n[t][agent][arm]
                        x[t+1][agent][arm] = x[t][agent][arm]
        
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

                

    
