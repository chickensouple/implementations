import math

def heuristic_none(node):
    return 0

def map_heuristic_l1(node, goal):
    h = abs(node[0]-goal[0]) + abs(node[1]-goal[1])
    return h

def map_heuristic_l2(node, goal):
    h = math.sqrt(abs(node[0]-goal[0])**2 + abs(node[1]-goal[1])**2)
    return h
