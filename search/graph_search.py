import queue
from heuristics import cost_heuristic_none, tie_heuristic_none, tie_heuristic_high_g
import heapq

class OpenList(object):
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



def astar(graph, start, goal, cost_heuristic, tie_heuristic=tie_heuristic_high_g):
    # add start node to open list
    frontier = OpenList()
    g = 0
    h = cost_heuristic(start)
    frontier.put(start, (g+h, tie_heuristic(g, h)))
    
    # set of nodes that have already been explored
    explored = set()

    # dict mapping children to parent
    predecessors = dict()

    # dict mapping nodes to cost from start
    costs = dict()
    costs[start] = 0

    
    path_found = False
    nodes_expanded = 0
    while not frontier.empty():
        node = frontier.get()
        nodes_expanded += 1

        # break if goal is found
        if node == goal:
            path_found = True
            break

        explored.add(node)

        # expand neighbors
        for neighbor, cost in zip(*graph.get_neighbors(node)):
            # if we have already explored neighbor dont add to open list
            if neighbor in explored:
                continue

            g = costs[node] + cost
            h = cost_heuristic(neighbor)
            priority = (g+h, tie_heuristic(g, h))

            # if frontier already has neighbor,
            # and priority is lower than what is already there
            # update the priority, otherwise, skip
            if frontier.contains(neighbor):
                if priority < frontier.get_priority(neighbor):
                    frontier.decrease_key(neighbor, priority)
                else:
                    continue
            else:
                # if frontier doesn't have neighbor, add to frontier
                frontier.put(neighbor, priority)

            # update cost from start and predecessor
            costs[neighbor] = g
            predecessors[neighbor] = node

    if not path_found:
        return path_found, [], None, nodes_expanded

    # construct path
    path = []
    if path_found:
        node = goal
        path.append(goal)
        while node != start:
            node = predecessors[node]
            path.append(node)
    path = path[::-1] # reverse list

    return path_found, path, costs[goal], nodes_expanded


def dijkstra(graph, start, goal, **kwargs):
    return astar(graph, start, goal, cost_heuristic_none, **kwargs)


if __name__ == '__main__':
    q = MinHeap()
    q.put('a', 1)
    q.put('b', -1)
    q.put('c', 2)
    q.put('d', 0)

    print(q.heap)
    q.decrease_key('c', -2)
    print(q.heap)


