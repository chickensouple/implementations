import math
import numpy as np

###################
# cost heuristics #
###################
def cost_heuristic_none(node):
    return 0

def cost_heuristic_l1(node, goal):
	h = np.linalg.norm(goal - node, ord=1)
    return h

def cost_heuristic_l2(node, goal):
	h = np.linalg.norm(goal - node, ord=2)
    return h

def cost_heuristic_linf(node, goal):
	h = np.linalg.norm(goal - node, ord=np.Inf)
	return h

###########################
# tie breaking heuristics #
###########################
def tie_heuristic_none(g_cost, h_cost):
	return g_cost + h_cost

def tie_heuristic_low_g(g_cost, h_cost):
	return g_cost

def tie_heuristic_high_g(g_cost, h_cost):
	return -g_cost


