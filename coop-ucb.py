# Simulator for the coop-ucb and coop-ucb2 algorithms in Landgren's papers

import random
import networkx as nx

# Global vars are simulation parameters
NUM_AGENTS = None  # M
NUM_ARMS = None  # N
true_means = []  # m_i
std_dev = None  # sigma_s
MAX_STEPS = None  # T
ucb2_flag = False
step_size = None  # kappa
gamma = 1

G = nx.Graph()


# gather user input to initialize network and simulation settings.
def initialize():
    print("Please take a moment to provide settings for your cooperative,"
          " distributed Multi-Armed Bandit problem. Settings include: \n"
          "- Number of agents\n"
          "- Agent network graph\n"
          "- Number of arms\n"
          "- Arm means (manually or randomly generated)\n"
          "- Standard deviation (manually or randomly generated)\n"
          "- Number of iterations\n"
          "- coop-ucb vs coop-ucb2\n"
          "- Step size parameter (optional)\n"
          "- Gamma (optional)")

    global NUM_AGENTS
    global NUM_ARMS
    global true_means
    global std_dev
    global MAX_STEPS
    global ucb2_flag
    global step_size
    global gamma

    print("Enter the number of agents (nodes): ")
    NUM_AGENTS = int(input())  # M
    if NUM_AGENTS <= 0:
        print("Error: Must have a positive integer of agents")
        exit()

    print("Would you like to Manually [M] or Randomly [R] generate a network graph of agents? ")
    graph_generation_method = input().capitalize()
    if graph_generation_method == "M":
        print("?") // TODO
    elif graph_generation_method == "R":
        print("?") // TODO
    else:
        print("Error: Must press either 'M' or 'R'")
        exit() 

    print("Enter the number of arms: ")
    NUM_ARMS = int(input())
    if NUM_ARMS <= 0:
        print("Error: Must have a positive integer of arms")
        exit()

    print("Would you like to Manually [M] set arm means or Randomly [R] generate arm means? ")
    mean_generation_method = input().capitalize()
    if mean_generation_method == "M":
        print("Please enter " + str(NUM_ARMS) + " means, separated by commas: ")
        true_means = [float(x) for x in input().split(",")]
    elif mean_generation_method == "R":
        print("Please enter a lower bound for the arm means: ")
        mean_lower_bound = float(input())
        print("Please enter an upper bound for the arm means: ")
        mean_upper_bound = float(input())
        if mean_lower_bound > mean_upper_bound:
            print("Error: lower bound greater than upper bound")
            exit()
        true_means = [random.uniform(mean_lower_bound, mean_upper_bound) for x in range(0, NUM_ARMS)]
    else:
        print("Error: Must press either 'M' or 'R'")
        exit()
    if len(true_means) != NUM_ARMS:
        print("Error. Must enter " + str(NUM_ARMS) + " means")
        exit()

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
            exit()
    elif sd_generation_method == "R":
        print("Please enter a lower bound for the standard deviation (must be nonnegative): ")
        sd_lower_bound = float(input())
        if sd_lower_bound < 0:
            print("Error: lower bound must be nonnegative")
            exit()
        print("Please enter an upper bound for the standard deviation (must be nonnegative): ")
        sd_upper_bound = float(input())
        if sd_upper_bound < 0:
            print("Error: upper bound must be nonnegative")
            exit()
        if sd_upper_bound < sd_lower_bound:
            print("Error: upper bound less than lower bound")
            exit()
        std_dev = random.uniform(sd_lower_bound, sd_upper_bound)
    else:
        print("Error: Must press either 'M' or 'R'")

    print("Enter the number of iterations for the simulation: ")
    MAX_STEPS = int(input())
    if MAX_STEPS <= 0:
        print("Error: Must have at least one iteration")
        exit()

    print("Press [2] if you would like to run the coop-ucb2 algorithm instead."
          " Press any other key for the original coop-ucb algorithm")
    if input() == "2":
        ucb2_flag = True

    print("Enter a step size parameter in the range (0,1]. If none chosen (just press ENTER),"
          " default is d_max / (d_max - 1), where d_max is the degree of the node with the most edges")
    step_size = input()
    if step_size == "":
        step_size = 1  # TODO: calculate d_max
    else:
        step_size = float(step_size)
        if step_size <= 0 or step_size > 1:
            print("Error: step size parameter must be in (0, 1]")
            exit()

    print("Enter a gamma parameter greater than or equal to 1. If none chosen (just press ENTER), default is 1")
    gamma = input()
    if gamma == "":
        gamma = 1
    else:
        gamma = float(gamma)
        if gamma < 1:
            print("Error: gamma must be greater than or equal to 1")
            exit()

    print("Done! Here are your settings:\n"
          "Number of agents: " + str(NUM_AGENTS) + "\n" +
          "Number of arms: " + str(NUM_ARMS) + "\n" +
          "Arm means: " + str(true_means) + "\n" +
          "Standard deviation: " + str(std_dev) + "\n" +
          "coop-ucb2?: " + str(ucb2_flag))
