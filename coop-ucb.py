# Simulator for the coop-ucb2 algorithm in Landgren's papers
# note: if running original coop-ucb, need eta parameter
# 10.1109/ECC.2016.7810293 <--- doi for coop-ucb original paper
# should provide a demo option!!
# issue # 1: not distributed
# issue # 2: what is gamma? Why is gamma = 1 in their simulation when paper says gamma > 1?
# issue # 3: kappa should be in (0, 1], but kappa = d_max / (d_max - 1) ensures greater than 1
# TODO: read more on gossip algorithms

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
    true_means = []
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
        true_means = [float(x) for x in input().split()]
    elif mean_generation_method == "R":
        print("Please enter a lower bound for the arm means: ")
        mean_lower_bound = float(input())
        print("Please enter an upper bound for the arm means: ")
        mean_upper_bound = float(input())
        if mean_lower_bound > mean_upper_bound:
            print("Error: lower bound greater than upper bound")
            return None
        true_means = [random.uniform(mean_lower_bound, mean_upper_bound) for x in range(0, num_arms)]
    else:
        print("Error: Must press either 'M' or 'R'")
        return None
    if len(true_means) != num_arms:
        print("Error. Must enter " + str(num_arms) + " means")
        return None
    
    return true_means


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
Returns True if we want to run coop-ucb2, False otherwise
*** NOT USED YET - focusing on coop-ucb2
'''
def get_coop_ucb_version():
    print("Press [1] if you would like to run the original coop-ucb algorithm instead."
        " Press any other key for coop-ucb2")
    return input().strip() != "1"


'''
Returns an instance of the distributed cooperative multi-armed bandit problem as specified by 
Landgren et al., in a tuple of the form (G, Mu, sigma, T, kappa, gamma). G is the network graph of agents 
(number of agents M is implicitly stored by G), Mu is a list of the true arm means (number of arms N is 
implicitly stored in Mu), sigma is the standard deviation, T is the number of time steps, kappa is the 
step size parameter, and gamma is some parameter I have yet to understand.
'''
def get_problem_instance():
    print("Please take a moment to provide problem parameters. Parameters include: \n"
    "- Agent network graph (manually or randomly generated)\n"
    "- Number of arms\n"
    "- Arm means (manually or randomly generated)\n"
    "- Standard deviation (manually or randomly generated)\n"
    "- Number of iterations\n"
    # "- coop-ucb vs coop-ucb2\n" <-- going to focus on coop-ucb2
    "- Step size parameter (default exists)\n"
    "- Gamma (default exists)")

    # get simulation parameters
    G = get_graph() # G
    true_means = get_arm_means() # Mu - list of m_i's
    std_dev = get_std_dev()  # sigma
    max_time_step = get_max_time_step(len(true_means))  # T
    step_size = get_step_size(G)  # kappa
    gamma = get_gamma() # all we know about this is supposed to be greater than 1 but they set to 1

    A = nx.adjacency_matrix(G).todense() # for printing purposes
    print("Here's what we got:\n"
    "Network Graph:\n\n " + str(A) + "\n\n"
    "Number of arms: " + str(len(true_means)) + "\n"
    "Arm means: " + str(true_means) + "\n"
    "Standard deviation: " + str(std_dev) + "\n"
    "Number of iterations: " + str(max_time_step) + "\n"
    "Step size parameter: " + str(step_size) + "\n"
    "Gamma: " + str(gamma))

    return G, true_means, std_dev, max_time_step, step_size, gamma


'''

'''
def coop_ucb2(G, true_means, std_dev, max_time_step, step_size, gamma):
    # initialize some variables
    num_agents = G.number_of_nodes() # M
    num_arms = len(true_means) # N
    count_per_agent_est = np.zeros((num_agents, num_arms)) # matrix of n_i^k(t)'s -- M x N
    tot_rwd_per_agent_est = np.zeros((num_agents, num_arms)) # matrix of s_i^k(t)'s -- M x N
    mean_per_agent_est = np.zeros((num_agents, num_arms)) # matrix of mu_i^k(t)'s -- M x N
    upper_confidence_bounds = np.zeros((num_agents, num_arms)) # matrix of Q_i^k(t)'s -- M x N


    






print("Welcome to the coop-ucb simulator. The simulator runs the coop-ucb2 algorithm as described in "
    "Landgren, Srivastava, and Leonard's paper. doi: 10.1109/CDC.2016.7798264")

get_problem_instance()
