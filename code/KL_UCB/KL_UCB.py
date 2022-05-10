import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt


class KL_UCB:
    ''' Representation of a single agent bandit problem and a method to run the KL_UCB algorithm on this problem

        Attributes
        ----------
        T: The number of time steps the KL_UCB algorithm will run for.
        arm_distributions: A list of scipy.stats probability distributions bounded on [0,1]
        means: A list of arm means. Extracted from arm_distributions
        M: Number of arms. Extracted from length of arm_distributions
        regret: A 1xT numpy ndarray of the expected regret from the most recent algorithm run
    '''

    def __init__(self, T, arm_distributions):
        ''' Construct a single agent bandit problem instance 
        
            Parameters
            ----------
            T: The number of time steps the KL_UCB algorithm will run for.
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

    def KL(self, p, q):
        ''' Compute Kullback-Leibler divergence between Bernoulli distributions of parameters p and q. 
        
            Note that we can use this even when arm distributions are not Bernoulli. In this case,
            p and q are the means of their respective distributions.
        '''
        if (p == 0 and q == 0) or (p == 1 and q == 1) or p == 0:
            return 0
        elif q == 0 or q == 1:
            return np.inf
        else:
            return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

    def dKL(self, p, q):
        ''' Compute the derivative of the Bernoulli Kullback-Leibler divergence between p and q, with respect to q. '''
        return (p-q)/(q*(q - 1.0))

    def newton(self, N, S, k, t, precision = 1e-3, max_iterations = 100, epsilon=1e-6):
        ''' Calculate upper confidence bound via newton's method 
        
            WARNING: This function works in that it efficiently finds greatest approx zero to f in (0,1).
            However, KL-UCB technically specifies that the returned value of q should be such that 
            f(q) <= 0. The q's returned by this function seemingly always satisfy f(q)>=0.
            Enforcing f(q) <= 0 does not work (times out) because f(q) converges to 0 from the right.
            If this is unacceptable, maybe look into other root finding methods like bisection.

            Parameters
            ----------
            N: A numpy ndarray holding the number of times each arm has been played up until time t.
            S: A numpy ndarray holding the cumulative reward received from each arm up until time t.
            k: The arm index we are computing an upper confidence bound for.
            t: The current time step.
            precision: Arbitrarily small convergence threshold
            max_iterations: Limit on number of iterations Newton's method should run
            epsilon: A miscellaneous arbitrarily small limit

            Return
            ------
            An upper confidence bound for the given parameters. Should technically be within [0,1], see warning.
        ''' 
        
        p = S[k]/N[k] # from paper
        delta = 0.1 # arbitrary small positive offset
        q = p + delta # initial guess. if p==q, then dKL=0 and we never move anywhere
        converged = False
        
        for n in range(max_iterations):
            if (p / q <= 0 or (1-p)/(1-q) <= 0): # sanity check for log domain errors
                print(f'log error: p={p}, q={q}, n={n}')
            # wish to find greatest val of q in (0,1) that gets f closest to zero from below
            f = self.KL(p, q) - np.log(t)/N[k]
            df = self.dKL(p, q) # derivative of f is just derivative of KL
            if abs(df) < epsilon: # check denominator not too small
                break 
            # NOTE: graph KL function to see that largest zero (what we want) is >= p.
            qnew = max(p+delta, min(q - (f / df), 1-epsilon)) # chris: my approach for keeping q in (p,1)
            # print(f'q={q}, f(q)={f}, qnew={qnew}, precision={precision} n={n}')
            if(abs(f) < precision and abs(qnew - q) < precision): # check for early convergence
                converged = True
                # print(f'converged at {n} iterations')
                break
            q = qnew    
        # if(not converged): print("Did not converge") 
        return q

    def plot_regret(self):
        ''' Plots regret of last run vs theoretical regret bounds 

            Note: make sure KL_UCB.run() was called before calling this method
        '''
        optimal_arm = np.argmax(self.means)
        time_axis = list(range(self.T))
        coeff = 0
        for i in range(self.M):
            if (i != optimal_arm): coeff += (self.means[optimal_arm] - self.means[i]) / (self.KL(self.means[i], self.means[optimal_arm]))
        theoretical_regret_bounds = [coeff * np.log(t+1) for t in time_axis] # not sure if allowed to do this bc of lim sup, seems like it works tho
        plt.plot(time_axis, theoretical_regret_bounds, '--')
        plt.plot(time_axis, self.regret)
        plt.show()

    def run(self):
        ''' Run the KL_UCB algorithm on the bandit problem instance held by self

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
            a = np.argmax([self.newton(N, S, arm, t) for arm in range(self.M)]) #argmax part of line 6 of algorithm 1
            r = self.arm_distributions[a].rvs()
            N[a] = N[a] + 1
            S[a] = S[a] + r
            exp_cum_rwd[t] = exp_cum_rwd[t-1] + self.means[a]
        optimal_exp_cum_rwd = [(t+1) * self.means[optimal_arm] for t in range(T)]
        regret = np.asarray(optimal_exp_cum_rwd) - np.asarray(exp_cum_rwd) # see definition of regret
        self.regret = regret
        return regret


# # test run
# T = 1000
# rwd_means = [.2, .3, .4, .5, .6]
# distributions = [sps.uniform(loc=rwd_means[i] - .1, scale=0.2) for i in range(len(rwd_means))]
# kl = KL_UCB(T, distributions)
# kl.run()
# kl.plot_regret()


