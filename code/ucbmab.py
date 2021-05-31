# Stationary single-agent MAB implementation of UCB algorithm
import numpy as np
import math
import matplotlib.pyplot as plt

# prompt user for number of arms, expected reward per arm, standard dev per arm, and number of iterations
print("Enter number of arms")
NUM_ARMS = int(input())
print("Enter " + str(NUM_ARMS) + " expected values, separated by commas")
str_averages = input().split(",")
mu = []
for s in str_averages:
    mu.append(float(s))
print("Enter " + str(NUM_ARMS) + " standard deviations, separated by commas")
str_sigmas = input().split(",")
sigmas = []
for s in str_sigmas:
    sigmas.append(float(s))
print("Enter number of iterations")
MAX_STEPS = int(input())
# now mu holds expected values, NUM_ARMS holds number of arms, sigmas holds std devs, MAX_STEPS holds T

# generates a reward from given arm
def fetch_reward(index):
    return np.random.normal(mu[index], sigmas[index])

# calculates upper confidence bound for a given time and arm
def ucb(i, t):
    if frq[i] == 0:
        return 1e400
    return running_avg[i] + math.sqrt(2 * math.log(t) / frq[i])

# total_rewards just used for plotting purposes
total_rewards = [0]

frq = [0] * NUM_ARMS
running_avg = [0] * NUM_ARMS

# proceed with ucb algorithm
for t in range(0, MAX_STEPS):
    opt_arm = 0
    for i in range(0, NUM_ARMS): # determine the optimal arm to choose this iteration
        if ucb(i, t) > ucb(opt_arm, t):
            opt_arm = i
    frq[opt_arm] += 1
    rwd = fetch_reward(opt_arm)
    total_rewards.append(rwd + total_rewards[t]) # for plotting purposes
    running_avg[opt_arm] = (running_avg[opt_arm] * (frq[opt_arm] - 1) + rwd) / frq[opt_arm]

# check how well we did by comparing total_reward against expected value for optimal
opt = max(mu) * MAX_STEPS # expected value if we had just picked distribution with highest mean every time
avg_rewards = []
for i in range(1, len(total_rewards)):
    avg_rewards.append(total_rewards[i] / i)
x_axis = list(range(0, MAX_STEPS))
plt.plot(x_axis, avg_rewards)
plt.show()
print(max(avg_rewards))







