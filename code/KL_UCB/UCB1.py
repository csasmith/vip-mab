import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt


class UCB1:
    ''' Representation of a single agent bandit problem and a method to run the UCB1 algorithm on this problem

        Attributes
        ----------
        T: The number of time steps the UCB1 algorithm will run for.
        arm_distributions: A list of scipy.stats probability distributions bounded on [0,1]
        means: A list of arm means. Extracted from arm_distributions
        M: Number of arms. Extracted from length of arm_distributions
        regret: A 1xT numpy ndarray of the expected regret from the most recent algorithm run
    '''

    def __init__(self, T, arm_distributions):
        ''' Construct a single agent bandit problem instance 
        
            Parameters
            ----------
            T: The number of time steps the UCB1 algorithm will run for.
            arm_distributions: A list of scipy.stats probability distributions bounded on [0,1].

            Raises
            ------
            ValueError
                If T is not a positive integer.
                If the support for any arm distribution is not in [0,1].
        '''
        if (T < 1 or type(T) is not int):
            raise ValueError('T must be a positive integer')
        if (any(d.support()[0] < 0 or d.support()[1] > 1 for d in arm_distributions)): 
            raise ValueError('distribution support must lie in [0,1]')
        self.T = T
        self.arm_distributions = arm_distributions
        self.means = [d.mean() for d in arm_distributions]
        self.M = len(arm_distributions)
        self.regret = None

    def C(self, n, t):
        ''' Calculate confidence width at time t for an arm pulled n times so far '''
        return np.sqrt((2 * np.log(t)) / n)

    def plot_regret(self):
        ''' Plots regret of last run vs theoretical regret bounds 

            Note: make sure UCB1.run() was called before calling this method
        '''
        optimal_arm = np.argmax(self.means)
        time_axis = list(range(self.T))
        # See Theorem 1 of Auer 2002
        gaps = [self.means[optimal_arm] - mean for mean in self.means]
        sum_gaps = np.sum(gaps)
        sum_gap_reciprocals = 0
        for gap in gaps:
            if (gap != 0):
                sum_gap_reciprocals += 1 / gap
        theoretical_regret_bounds = [8 * np.log(t+1) * sum_gap_reciprocals + (1 + (np.pi ** 2)/3) * sum_gaps  for t in time_axis]
        plt.plot(time_axis, theoretical_regret_bounds, '--')
        plt.plot(time_axis, self.regret)
        plt.show()

    def run(self):
        ''' Run the UCB1 algorithm on the bandit problem instance held by self

            Return
            ------
            A list of length self.T with expected regret at each time t

        '''
        N = np.zeros(self.M) # keeps track of number of times arm k has been chosen
        S = np.zeros(self.M) # keeps track of cumulative sum of rewards for arm k
        # data structures just for plotting regret
        optimal_arm = np.argmax(self.means)
        exp_cum_rwd = [0 for t in range(self.T)] # exp_cum_rwd[t] is expected cumulative reward at time t
        for t in range(self.M):
            N[t] = 1
            S[t] = self.arm_distributions[t].rvs()
            exp_cum_rwd[t] = exp_cum_rwd[t-1] + self.means[t] if t != 0 else self.means[t] # t is index of chosen arm here
        for t in range(self.M, self.T):
            a = np.argmax([(S[arm]/N[arm]) + self.C(N[arm], t) for arm in range(self.M)])
            r = self.arm_distributions[a].rvs()
            N[a] = N[a] + 1
            S[a] = S[a] + r
            exp_cum_rwd[t] = exp_cum_rwd[t-1] + self.means[a]
        optimal_exp_cum_rwd = [(t+1) * self.means[optimal_arm] for t in range(self.T)]
        regret = np.asarray(optimal_exp_cum_rwd) - np.asarray(exp_cum_rwd) # see definition of regret
        self.regret = regret
        return regret


# # test run
# T = 100000
# rwd_means = [.2, .3, .4, .5, .6]
# sd = 0.5
# distributions = [sps.truncnorm(a=(0 - rwd_means[i]) / sd, b=(1 - rwd_means[i]) / sd, loc=rwd_means[i], scale=0.2) for i in range(len(rwd_means))]
# kl = UCB1(T, distributions)
# kl.run()
# kl.plot_regret()


