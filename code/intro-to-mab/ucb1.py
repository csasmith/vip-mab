# UCB1 implementation and simulator
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

'''
Input vars:
    - number of arms K
    - number of time steps T
    - K probability distributions (bounded in [0,1]), need their means, standard devs, etc.
'''

K = 3
T = 10000
# how we set up distributions depends on which ones we want to use - i.e. this would be different for bernoulli
# going to stick with normal distributions for now 
# (truncnorm so that support is [0,1]. Notice still works well with non-truncated normal distribution...)
# for more info on truncnorm, see:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html#scipy.stats.truncnorm
# for info on how to actually sample from one of these scipy distributions, see
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous
means = [0.2, 0.3, 0.7] # make sure the length of this list matches K
std_devs = [0.3, 0.2, 0.4]
distributions = []
for mean, sd in zip(means, std_devs):
    a = (0 - mean) / sd
    b = (1- mean) / sd
    distributions.append(sps.truncnorm(a, b, loc=mean, scale=sd))

'''
Data structures & variables (what do we need to keep track of?)
    * For actually running the algorithm:
        - list of length K for the running average reward of each arm
        - list of length K for the number of times each arm has been pulled
    * For plotting regret:
        - index of best arm
        - list for index of what arm is chosen at each time step
'''

# For algorithm
running_avg_rewards = [0 for i in range(K)]
num_pulls = [0 for i in range(K)]
# For plotting
optimal_arm = np.argmax(means)
my_expected_cumulative_reward = [0 for t in range(T)]

'''
Helper methods
'''

def calc_ucb(i: int, t: int) -> float:
    '''
    Calculate the upper confidence bound for arm i at time t
    '''
    avg_rwd = running_avg_rewards[i]
    n_i = num_pulls[i]
    confidence_width = np.sqrt((2 * np.log(t)) / n_i)
    ucb = avg_rwd + confidence_width
    return ucb

'''
Main loop
'''

for t in range(T):
    if (t < K): # sample each arm once
        running_avg_rewards[t] = distributions[t].rvs(1) # take one sample from this distribution
        num_pulls[t] += 1
        my_expected_cumulative_reward[t] = my_expected_cumulative_reward[t-1] + means[t] if t != 0 else means[t]
    else:
        upper_confidence_bounds = [calc_ucb(i,t) for i in range(K)]
        chosen_arm = np.argmax(upper_confidence_bounds)
        reward = distributions[chosen_arm].rvs(1)
        old_avg = running_avg_rewards[chosen_arm]
        old_num_pulls = num_pulls[chosen_arm]
        num_pulls[chosen_arm] += 1
        running_avg_rewards[chosen_arm] = ((old_avg * old_num_pulls) + reward) / num_pulls[chosen_arm]
        my_expected_cumulative_reward[t] = my_expected_cumulative_reward[t-1] + means[chosen_arm] # for plotting

print("Done running UCB1!")

'''
Plotting time
'''

time_axis = list(range(T))

# plot curve for theoretical regret bounds in Theorem 1 of Auer 2002
gaps = [means[optimal_arm] - mean for mean in means]
sum_gaps = np.sum(gaps)
sum_gap_reciprocals = 0
for gap in gaps:
    if (gap != 0):
        sum_gap_reciprocals += 1 / gap
theoretical_regret_bounds = [8 * np.log(t+1) * sum_gap_reciprocals + (1 + (np.pi ** 2)/3) * sum_gaps  for t in time_axis]
plt.plot(time_axis, theoretical_regret_bounds, '--')

# plot our regret
optimal_expected_cumulative_reward = [means[optimal_arm] * (t+1) for t in range(T)]
expected_regret = np.asarray(optimal_expected_cumulative_reward) - np.asarray(my_expected_cumulative_reward)
plt.plot(time_axis, expected_regret)
plt.show()
