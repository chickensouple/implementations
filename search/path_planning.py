import queue
from heuristics import heuristic_none

import numpy as np
import matplotlib.pyplot as plt


import heapq


class ClosedList(object):
    def __init__(self):
        self.heap = []

        self.heap_dict = dict()

    def put(self, item, priority):
        heapq.heappush(self.heap, (priority, item))
        self.heap_dict[item] = priority

    def get(self):
        _, item = heapq.heappop(self.heap)
        self.heap_dict.pop(item)
        return item

    def empty(self):
        return len(self.heap) == 0

    def contains(self, item):
        return item in self.heap_dict

    def get_priority(self, item):
        return self.heap_dict[item]

    def decrease_key(self, item, new_priority):
        # linear search
        idx = None
        for i, val in enumerate(self.heap):
            if val[1] == item:
                idx = i
                break

        old_priority = self.heap[idx][0]
        self.heap[idx] = (new_priority, item)
        self.heap_dict[item] = new_priority
        heapq._siftdown(self.heap, 0, idx)


def astar(graph, start, goal, heuristic):
    # requires admissible heuristic
    frontier = ClosedList()
    explored = set()

    predecessors = dict()

    costs = dict()
    costs[start] = 0

    # frontier.put((0 + heuristic(start), start))
    frontier.put(start, 0 + heuristic(start))
    path_found = False
    while not frontier.empty():
        # curr_cost, node = frontier.get()
        node = frontier.get()
        
        if node in explored:
            continue

        if node == goal:
            path_found = True
            break

        explored.add(node)

        for neighbor in graph.get_neighbors(node):
            if neighbor in explored:
                continue

            cost = costs[node] + graph.get_cost(node, neighbor)
            priority = cost + heuristic(neighbor)

            if frontier.contains(neighbor):
                if priority < frontier.get_priority(neighbor):
                    frontier.decrease_key(neighbor, priority)
                else:
                    continue
            else:
                frontier.put(neighbor, priority)

            costs[neighbor] = cost
            predecessors[neighbor] = node

    if not path_found:
        return path_found, [], None

    # construct path
    path = []
    if path_found:
        node = goal
        path.append(goal)
        while node != start:
            node = predecessors[node]
            path.append(node)
    path = path[::-1] # reverse list

    return path_found, path, costs[goal]



def dijkstra(graph, start, goal):
    return astar(graph, start, goal, heuristic_none)




if __name__ == '__main__':
    q = MinHeap()
    q.put('a', 1)
    q.put('b', -1)
    q.put('c', 2)
    q.put('d', 0)

    print(q.heap)
    q.decrease_key('c', -2)
    print(q.heap)


