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

    def peek(self):
        return self.heap[0]

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


def astar(graph, start, goal, cost_heuristic=cost_heuristic_none, tie_heuristic=tie_heuristic_high_g):
    """
    Performs A* search on a graph with a given start and goal

    cost_heuristic is the guiding heuristic (h value) used in A*.
    By default, it is the zero function, which will just perform djikstra's algorithm
    tie_heuristic is a heuristic to break ties in the total f=g+h value.
    the priority of a node will be (f, tie_heuristic(g, h)). 
    So nodes, that have high tie_heuristic will be prioritized in the case of a tie in the f value.    
    
    Args:
        graph (object): object of baseclass type GraphBase
        start (TYPE): start node of search
        goal (TYPE): goal node of search
    
    Returns:
        tuple: (path_found, path, cost, nodes_expanded)
        path_found is a bool indicating whether search was successful
        path is a list of nodes from start to goal 
        cost is the cost of the path
        nodes_expanded is the number of nodes expanded during search
    """
    open_list = OpenList()

    # add start node to open list
    g = 0
    h = cost_heuristic(start)
    open_list.put(start, (g+h, tie_heuristic(g, h)))
    
    # set of nodes that have already been explored
    explored = set()

    # dict mapping children to parent
    predecessors = dict()

    # dict mapping nodes to cost from start
    costs = dict()
    costs[start] = 0

    path_found = False
    nodes_expanded = 0
    while not open_list.empty():
        node = open_list.get()
        nodes_expanded += 1

        # break if goal is found
        if node == goal:
            path_found = True
            break

        explored.add(node)

        # expand node
        for successor, cost in zip(*graph.get_successors(node)):
            # if we have already explored successor dont add to open list
            if successor in explored:
                continue

            g = costs[node] + cost
            h = cost_heuristic(successor)
            priority = (g+h, tie_heuristic(g, h))

            # if open_list already has successor,
            # and priority is lower than what is already there
            # update the priority, otherwise, skip
            if open_list.contains(successor):
                if priority < open_list.get_priority(successor):
                    open_list.decrease_key(successor, priority)
                else:
                    continue
            else:
                # if open_list doesn't have successor, add to open_list
                open_list.put(successor, priority)

            # update cost from start and predecessor
            costs[successor] = g
            predecessors[successor] = node

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


class LPAStar(object):
    def __init__(self, cost_heuristic=cost_heuristic_none):
        """
        Sets up basic options for LPA* search
        cost_heuristic is the guiding heuristic (h value) used in LPA*.
        By default, it is the zero function, which will just perform djikstra's algorithm

        Args:
            cost_heuristic (func, optional): function that takes in (current_node, goal_node) and returns a scalar
        """
        self.cost_heuristic = cost_heuristic
        self.reset()

    def reset(self):
        self.open_list = OpenList()
        self.rhs = dict()
        self.g = dict()

    def search(self, graph, start, goal):
        # set initial rhs values
        self.rhs[start] = 0
        self.rhs[goal] = float('inf')

        self.g[start] = float('inf')
        self.g[goal] = float('inf')

        # add start node to open list
        h = self.cost_heuristic(start)
        self.open_list.put(start, (h, 0))

        # print self.open_list.peek()
        nodes_expanded = 0
        while not self.open_list.empty() and (self.open_list.peek()[0] < self._compute_key(goal) or self.rhs[goal] != self.g[goal]):
            node = self.open_list.get()
            if self.g.get(node, float('inf')) > self.rhs.get(node, float('inf')):
                self.g[node] = self.rhs.get(node, float('inf'))
                for successor in graph.get_successors(node)[0]:
                    self._update_node(successor, start, graph)
            else:
                self.g[node] = float('inf')
                self._update_node(node, start, graph)
                for successor in graph.get_successors(node)[0]:
                    self._update_node(successor, start, graph)
            nodes_expanded += 1
        print("nodes expanded: " + str(nodes_expanded))


    def _update_node(self, node, start, graph):
        if node != start:
            predecessors, costs = graph.get_predecessors(node)
            possible_rhs_vals = [self.rhs.get(predecessor, float('inf'))+cost for predecessor, cost in zip(predecessors, costs)]
            self.rhs[node] = min(possible_rhs_vals)
        if self.open_list.contains(node):
            # remove from open list
            self.open_list.decrease_key(node, (-1, -1))
            self.open_list.get()
            # TODO: more efficient way to do this?
        if self.g.get(node, float('inf')) != self.rhs.get(node, float('inf')):
            self.open_list.put(node, self._compute_key(node))

    def _compute_key(self, node):
        g = self.g.get(node, float('inf'))
        rhs = self.rhs.get(node, float('inf'))

        min_val = min(g, rhs)
        key = (min_val + self.cost_heuristic(node), min_val)
        return key




if __name__ == '__main__':
    q = OpenList()
    q.put('a', 1)
    q.put('b', -1)
    q.put('c', 2)
    q.put('d', 0)

    print(q.heap)
    q.decrease_key('c', -2)
    print(q.heap)


