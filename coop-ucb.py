# Simulator for the coop-ucb and coop-ucb2 algorithms in Landgren's papers

import random
import networkx as nx

print("Welcome to the coop-ucb simulator. The simulator runs the coop-ucb algorithm as described in "
    "Landgren, Srivastava, and Leonard's papers: 10.1109/ECC.2016.7810293 and 10.1109/CDC.2016.7798264")
print("Please take a moment to provide problem parameters. Parameters include: \n"
    "- Agent network graph (manually or randomly generated)\n"
    "- Number of arms\n"
    "- Arm means (manually or randomly generated)\n"
    "- Standard deviation (manually or randomly generated)\n"
    "- Number of iterations\n"
    "- coop-ucb vs coop-ucb2\n"
    "- Step size parameter (optional)\n"
    "- Gamma (optional)")

# simulation parameters
G = get_graph() # G
true_means = get_arm_means() # m_i's
std_dev = get_std_dev()  # sigma_s
max_time_step = get_max_time_step()  # T
step_size = get_step_size()  # kappa
gamma = get_gamma() # all we know about this is supposed to be greater than 1 but they set to 1

# other options
ucb2_flag = get_coop_ucb_version() # coop-ucb2 easier (and better?)

# globals derived from parameters
num_agents = G.number_of_nodes() # N
num_arms = len(true_means) # M

# gather user input to initialize network and simulation settings.
def initialize():

    print("Done! Here are your settings:\n"
          "Number of agents: " + str(NUM_AGENTS) + "\n" +
          "Number of arms: " + str(NUM_ARMS) + "\n" +
          "Arm means: " + str(true_means) + "\n" +
          "Standard deviation: " + str(std_dev) + "\n" +
          "coop-ucb2?: " + str(ucb2_flag))


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
            for i in range(num_agents):
                edge_list.append(tuple(map(int, input().split())))
            g.add_edges_from(edge_list)
    elif graph_generation_method == "R":
        print("The random graph is generated as an erdos-renyi graph. Please enter the number of agents:")
        num_agents = int(input())
        print("Please enter a probability (ie 0.5")
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
'''
def get_max_time_step():
    print("Enter the number of iterations for the simulation: ")
    max_time_step = int(input())
    if max_time_step <= 0:
        print("Error: Must have at least one iteration")
        return None
    return max_time_step


'''
Returns True if we want to run coop-ucb2, False otherwise
'''
def get_coop_ucb_version():
    print("Press [1] if you would like to run the original coop-ucb algorithm instead."
        " Press any other key for coop-ucb2")
    return input().strip() != "1"


'''
Return step size parameter (kappa) in range (0, 1]. Default is d_max / (d_max - 1), 
where d_max is the degree of the node with the most edges
TODO: calculate default
'''
def get_step_size():
    print("Enter a step size parameter in the range (0,1]. If none chosen (just press ENTER),"
          " default is d_max / (d_max - 1), where d_max is the degree of the node with the most edges")
    step_size = input()
    if step_size == "":
        step_size = 1  # TODO: calculate d_max
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
