# Simulator for the coop-ucb2 algorithm in Landgren's paper doi: 10.1109/CDC.2016.7798264
# 10.1109/ECC.2016.7810293 <--- doi for coop-ucb original paper
# issue # 1: not actually distributed...
# issue # 2: what is gamma? Why is gamma = 1 in their simulation when paper says gamma > 1?
# issue # 3: kappa should be in (0, 1], but kappa = d_max / (d_max - 1) ensures greater than 1... so what gives?

import random
import networkx as nx
import numpy as np

''' 
Returns G: a networkx graph instance built from user input.
G can be generated manually or randomly. Random generation is accomplished
via an erdos-renyi graph. Manual generation is accomplished either by 
typing edge pairs to standard input, or providing the path for a file with graph data in
multiline adjacency list format. See networkx documentation for more details.
'''
def get_graph():
    g = nx.Graph()
    print("Would you like to Manually [M] or Randomly [R] generate a network graph of agents? ")
    graph_generation_method = input().capitalize()
    if graph_generation_method == "M":
        print("Press [F] to provide a file with graph data, or any other key to type in the graph")
        manual_generation_method = input().capitalize()
        if manual_generation_method == 'F':
            print("Enter the path to a file containing graph data in multiline adjacency list format. See networkx docs for details")
            g = nx.read_multiline_adjlist(input().strip())
        else:
            edge_list = []
            print("Please enter the number of agents:")
            num_agents = int(input())
            print("Enter an edge pair like '1 2' on each line for " + str(num_agents) + " lines")
            for _ in range(num_agents):
                edge_list.append(tuple(map(int, input().split())))
            g.add_edges_from(edge_list)
    elif graph_generation_method == "R":
        print("The random graph is generated as an erdos-renyi graph. Please enter the number of agents:")
        num_agents = int(input())
        print("Please enter a probability (ie 0.5)")
        p = float(input())
        if p < 0 or p > 1:
            print("invalid probability")
            return None
        g = nx.erdos_renyi_graph(num_agents, p)
    else:
        print("Error: Must press either 'M' or 'R'")
        return None 

    return g


''' 
Returns arm means as a list of m_i's, where m_i refers to the true mean of arm i. 
These are unknown to the algorithm, and are used for simulation purposes. Arm means are 
generated either randomly, or manually through standard input. Random arm mean generation 
requires that the user input lower and upper bounds for means. Means are then sampled from 
a uniform distribution given by these lower and upper bounds.
'''
def get_arm_means():
    arm_means = []
    # get number of arms
    print("Enter the number of arms: ")
    num_arms = int(input())
    if num_arms <= 0:
        print("Error: number of arms must be a positive integer")
        return None
    
    # get means
    print("Would you like to Manually [M] set arm means or Randomly [R] generate arm means? ")
    mean_generation_method = input().capitalize()
    if mean_generation_method == "M":
        print("Please enter " + str(num_arms) + " means, separated by whitespace: ")
        arm_means = [float(x) for x in input().split()]
    elif mean_generation_method == "R":
        print("Please enter a lower bound for the arm means: ")
        mean_lower_bound = float(input())
        print("Please enter an upper bound for the arm means: ")
        mean_upper_bound = float(input())
        if mean_lower_bound > mean_upper_bound:
            print("Error: lower bound greater than upper bound")
            return None
        arm_means = [random.uniform(mean_lower_bound, mean_upper_bound) for x in range(0, num_arms)]
    else:
        print("Error: Must press either 'M' or 'R'")
        return None
    if len(arm_means) != num_arms:
        print("Error. Must enter " + str(num_arms) + " means")
        return None
    
    return arm_means


'''
Returns sigma_s, where sigma_s refers to the standard deviation shared by each arm. Unlike the 
arm means, this value is known to the algorithm.
'''
def get_std_dev():
    std_dev = 0
    print("Would you like to Manually [M] set the standard deviation or Randomly [R] generate"
        " the standard deviation? ")
    sd_generation_method = input().capitalize()
    if sd_generation_method == "M":
        print("Please enter a nonnegative real number: ")
        raw_sd = float(input())
        if raw_sd >= 0:
            std_dev = raw_sd
        else:
            print("Error: standard deviation must be a nonnegative number")
            return None
    elif sd_generation_method == "R":
        print("Please enter a lower bound for the standard deviation (must be nonnegative): ")
        sd_lower_bound = float(input())
        if sd_lower_bound < 0:
            print("Error: lower bound must be nonnegative")
            return None
        print("Please enter an upper bound for the standard deviation (must be nonnegative): ")
        sd_upper_bound = float(input())
        if sd_upper_bound < 0:
            print("Error: upper bound must be nonnegative")
            return None
        if sd_upper_bound < sd_lower_bound:
            print("Error: upper bound less than lower bound")
            return None
        std_dev = random.uniform(sd_lower_bound, sd_upper_bound)
    else:
        print("Error: Must press either 'M' or 'R'")
        return None

    return std_dev


'''
Returns T, the number of iterations / time steps / "horizon length"
T must be greater than the number of agents, because the initalization
step of the coop-ucb algorithm(s) has each agent sample each arm once
'''
def get_max_time_step(num_arms):
    print("Enter the number of iterations for the simulation: ")
    max_time_step = int(input())
    if max_time_step < num_arms:
        print("Error: Must have at least as many iterations as there are arms")
        return None
    return max_time_step


'''
Return step size parameter (kappa) in range (0, 1]. Default is 0.5.
TODO: issue no.3: d_max / (d_max - 1) is greater than 1
'''
def get_step_size(G):
    print("Enter a step size parameter in the range (0,1]. If none chosen (just press ENTER),"
        " default is .5")
    step_size = input()
    if step_size == "":
        '''
        d_max = max(list(G.degree), key = lambda x:x[1])[1]
        step_size = d_max / (d_max - 1)
        '''
        step_size = 0.5
    else:
        step_size = float(step_size)
        if step_size <= 0 or step_size > 1:
            print("Error: step size parameter must be in (0, 1]")
            return None
    
    return step_size


'''
Return gamma parameter. TODO: figure out what it means
'''
def get_gamma():
    print("Enter a gamma parameter greater than or equal to 1. If none chosen (just press ENTER), default is 1")
    gamma = input()
    if gamma == "":
        gamma = 1
    else:
        gamma = float(gamma)
        if gamma < 1:
            print("Error: gamma must be greater than or equal to 1")
            return None

    return gamma


'''
Returns an instance of the distributed cooperative multi-armed bandit problem as specified by 
Landgren et al., in a tuple of the form (G, arm_means, std_dev, max_time_step, step_size, gamma). 
G is the network graph of agents (number of agents M is implicitly stored by G), arm_means is a 
list of the true arm means unknown to the algorithm (number of arms N is implicitly stored in arm_means), 
std_dev is the standard deviation shared by all the arms, max_time_step is the number of time steps, step_size is the 
step size parameter (kappa), and gamma is some parameter I have yet to understand.
'''
def get_problem_instance():
    print("Press [C] to choose your problem parameters, or any other key to see a demo")
    if input().capitalize() == 'C':
        print("Please take a moment to provide problem parameters. Parameters include: \n"
        "- Agent network graph (manually or randomly generated)\n"
        "- Number of arms\n"
        "- Arm means (manually or randomly generated)\n"
        "- Standard deviation (manually or randomly generated)\n"
        "- Number of iterations\n"
        # "- coop-ucb vs coop-ucb2\n" <-- going to focus on coop-ucb2
        "- Step size parameter (default exists)\n"
        "- Gamma (default exists)\n")

        # get simulation parameters
        G = get_graph() # G
        arm_means = get_arm_means() # Mu - list of m_i's
        std_dev = get_std_dev()  # sigma
        max_time_step = get_max_time_step(len(arm_means))  # T
        step_size = get_step_size(G)  # kappa
        gamma = get_gamma() # all we know about this is supposed to be greater than 1 but they set to 1
    else:
        G = nx.bull_graph()
        arm_means = [1, 3, 5, 5]
        std_dev = 1
        max_time_step = 100
        step_size = 0.5
        gamma = 1

    A = nx.adjacency_matrix(G).todense() # for printing purposes
    print("PARAMETERS:\n\n"
    "Network Graph Adjacency Matrix:\n\n " + str(A) + "\n\n"
    "Number of arms: " + str(len(arm_means)) + "\n"
    "Arm means: " + str(arm_means) + "\n"
    "Standard deviation: " + str(std_dev) + "\n"
    "Number of iterations: " + str(max_time_step) + "\n"
    "Step size parameter: " + str(step_size) + "\n"
    "Gamma: " + str(gamma) + "\n")

    return G, arm_means, std_dev, max_time_step, step_size, gamma


'''
Returns an M x N matrix of C_i^k(t)'s, where C_i^k(t) is the upper confidence bound of agent k for 
arm i at time t. Parameters include n_t, t, gamma, sd, and M, where n_t is the matrix of the agent's 
estimates of the total number of times each arm has been pulled per unit agent, t is the current time, 
gamma is the same gamma parameter coop-ucb2 takes, sd is the standard deviation, and M is the number 
of agents. We use sqrt(ln(t)) as the sublogarithmic function f(t)
'''
def compute_ucb(n_t, t, gamma, sd, M):
    # print(type(n_t))
    # print("original n_t shape: " + str(n_t.shape))
    ucb = np.log(t) / n_t
    # print("np.log(t) / n_t shape: " + str(ucb.shape))
    numerator = n_t + np.sqrt(np.log(t))
    # print("numerator shape: " + str(numerator.shape))
    denom = M * n_t
    # print("denom shape: " + str(denom.shape))
    frac = numerator / denom
    # print("frac shape: " + str(frac.shape))
    # print(type(frac))
    # print(type(ucb))
    ucb = frac * ucb
    ucb = 2 * gamma * ucb
    ucb = np.sqrt(ucb)
    ucb = sd * ucb
    # sd * np.sqrt(2 * gamma * ((n_t + np.sqrt(np.log(t))) / (M * n_t)) * (np.log(t) / n_t))
    return ucb


'''
Runs the coop_ucb2 algorithm on a problem instance as given by get_problem_instance().
'''
def coop_ucb2(G, arm_means, std_dev, max_time_step, step_size, gamma):
    
    # initialize variables and perform initialization step of algorithm

    # important constants
    num_agents = G.number_of_nodes() # M
    num_arms = len(arm_means) # N

    # used only for tracking algorithm performance
    # opt_mean = max(arm_means) # m_i*
    # think with just arm_history we can calculate expected cumulative regret at very end
    arm_history = np.array([[i for i in range(num_arms)] for _ in range(num_agents)]) # i^k(t) -- M x T

    # for cooperative estimation step
    d_max = max(list(G.degree), key=lambda x:x[1])[1] # max degree of any node in G
    # NOTE: needed to call toarray() on laplacian or else everything gets converted to matrix instead of array and breaks
    P = np.eye(num_agents) - (step_size / d_max) * nx.laplacian_matrix(G).toarray() # row stochastic matrix P -- M x M
    
    # matrices that are continuously read/written, ie the important stuff
    arms_picked_indicators = np.zeros((num_agents, num_arms)) # matrix of zeta_i^k(t)'s -- M x N
    rwds_this_iteration = np.zeros((num_agents, num_arms)) # matrix of r_i^k(t)'s -- M x N
    est_cnt_per_agent = np.ones((num_agents, num_arms)) # matrix of n_i^k(t)'s -- M x N
    est_tot_rwd_per_agent = np.random.normal(loc=arm_means, scale=std_dev, size=(num_agents, num_arms)) # matrix of s_i^k(t)'s -- M x N
    est_mean_rwd_per_agent = est_tot_rwd_per_agent.copy() # matrix of mu_i^k(t)'s -- M x N
    upper_confidence_bounds = np.zeros((num_agents, num_arms)) # matrix of Q_i^k(t)'s -- M x N

    # at this point the initialization step of the algorithm is done - 
    # notice how we initialized some of the variables

    # main loop for each time step
    for t in range(num_agents, max_time_step + 1):
        # compute ucb, pick arm and get reward
        upper_confidence_bounds = compute_ucb(est_cnt_per_agent, t, gamma, std_dev, num_agents)
        arms_picked = np.argmax(est_mean_rwd_per_agent + upper_confidence_bounds, axis=1)
        arm_history = np.c_[arm_history, arms_picked] # for tracking performance, append column of arms picked
        for i in range(num_agents):
            arms_picked_indicators[i, arms_picked[i]] = 1 # update zeta_i^k(t)'s
        rwds_this_iteration = np.random.normal(loc=arm_means, scale=std_dev, size=(num_agents, num_arms)) * arms_picked_indicators

        # perform cooperative estimation -- we can just do P times n(t) matrix, etc. shapes work out nicely
        est_cnt_per_agent = P @ est_cnt_per_agent + P @ arms_picked_indicators
        est_tot_rwd_per_agent = P @ est_tot_rwd_per_agent + P @ rwds_this_iteration
        est_mean_rwd_per_agent = est_tot_rwd_per_agent / est_cnt_per_agent

    return est_cnt_per_agent, est_tot_rwd_per_agent, est_mean_rwd_per_agent, arm_history


print("Welcome to the coop-ucb simulator. The simulator runs the coop-ucb2 algorithm as described in "
    "Landgren, Srivastava, and Leonard's paper. doi: 10.1109/CDC.2016.7798264\n")

instance = get_problem_instance()
results = coop_ucb2(*instance)

# now do performance / regret analysis

# get some vars
arm_means = instance[1]
max_time_step = instance[3]
est_cnt_per_agent = results[0]
est_tot_rwd_per_agent = results[1]
est_mean_rwd_per_agent = results[2]
# contains indices into arm_means for each agent for each choice at each time step, ie m_i^k(t)'s
arm_history = results[3]

print("RESULTS: \n")
print("Estimated number of times each arm was sampled per unit agent: \n" + str(est_cnt_per_agent) + "\n")
print("Estimated total reward for each arm per unit agent: \n" + str(est_tot_rwd_per_agent) + "\n")
print("Estimated mean reward for each arm per unit agent: \n" + str(est_mean_rwd_per_agent) + "\n")

exp_opt_rwd = np.ones(arm_history.shape) * max(arm_means)
exp_cum_opt_rwd = np.cumsum(exp_opt_rwd, axis=1)
exp_actual_rwd = np.zeros(arm_history.shape)

for i in range(arm_history.shape[0]):
    for j in range(arm_history.shape[1]):
        exp_actual_rwd[i, j] = arm_means[arm_history[i, j]]

exp_cum_actual_rwd = np.cumsum(exp_actual_rwd, axis=1)
exp_cum_regret = exp_cum_opt_rwd - exp_cum_actual_rwd

print("Regret Analysis:")
print("The best possible expected cumulative reward (all agents choose optimal arm every time) was: " + str(np.sum(exp_cum_opt_rwd.T[-1])))
print("The total expected cumulative reward was: " + str(np.sum(exp_cum_actual_rwd.T[-1])))
print("The expected cumulative regret was: " + str(np.sum(exp_cum_regret.T[-1])))

# TODO: matplotlib it 











