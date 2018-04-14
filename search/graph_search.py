import queue
from heuristics import cost_heuristic_none, tie_heuristic_none, tie_heuristic_high_g
import heapq
import numpy as np
import time
import copy

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
    def remove(self, item):
        self.decrease_key(item, (-float('inf'), -float('inf')))
        self.get()

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
            # if we have already explored successor don't add to open list
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
        return path_found, [], float('inf'), nodes_expanded

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
    def __init__(self, graph, start, goal, cost_heuristic=cost_heuristic_none):
        """
        Sets up basic options for LPA* search
        cost_heuristic is the guiding heuristic (h value) used in LPA*.
        By default, it is the zero function, which will just perform djikstra's algorithm

        Args:
            cost_heuristic (func, optional): function that takes in (current_node, goal_node) and returns a scalar
        """
        self.cost_heuristic = cost_heuristic
        self.graph = graph
        self.start = start
        self.goal = goal
        self.reset()

    def reset(self):
        self.open_list = OpenList()
        self.rhs = dict()
        self.g = dict()

        # set initial rhs values
        self.rhs[self.start] = 0
        self.rhs[self.goal] = float('inf')

        self.g[self.start] = float('inf')
        self.g[self.goal] = float('inf')

        # add start node to open list
        h = self.cost_heuristic(self.start)
        self.open_list.put(self.start, (h, 0))


    def update_graph(self, graph):
        self.graph = graph

    def search(self):
        # node expansions
        nodes_expanded = 0
        while not self.open_list.empty() and \
            (self.open_list.peek()[0] < self._compute_key(self.goal) or self.rhs[self.goal] != self.g[self.goal]):

            node = self.open_list.get()
            if self.g.get(node, float('inf')) > self.rhs.get(node, float('inf')):
                self.g[node] = self.rhs.get(node, float('inf'))
                for successor in self.graph.get_successors(node)[0]:
                    self._update_node(successor)
            else:
                self.g[node] = float('inf')
                self._update_node(node)
                for successor in self.graph.get_successors(node)[0]:
                    self._update_node(successor)
            nodes_expanded += 1


        # path not found
        if self.rhs[self.goal] != self.g[self.goal]:
            return False, [], float('inf'), nodes_expanded

        # back out path
        node = self.goal
        path = [self.goal]
        while node != self.start:
            predecessors, _ = self.graph.get_predecessors(node)
            vals = np.array([self.g.get(pred, float('inf')) for pred in predecessors])
            idx = np.argmin(vals)
            node = predecessors[idx]
            path.append(node)
        path.append(self.start)
        path.reverse()

        return True, path, self.g[self.goal], nodes_expanded

    def _update_node(self, node):
        if node != self.start:
            predecessors, costs = self.graph.get_predecessors(node)
            possible_rhs_vals = [self.rhs.get(pred, float('inf'))+cost for pred, cost in zip(predecessors, costs)]
            if len(possible_rhs_vals) == 0:
                # no possible rhs vals
                self.rhs[node] = float('inf')
            else:
                self.rhs[node] = min(possible_rhs_vals)
        if self.open_list.contains(node):
            self.open_list.remove(node)
        if self.g.get(node, float('inf')) != self.rhs.get(node, float('inf')):
            self.open_list.put(node, self._compute_key(node))

    def _compute_key(self, node):
        g = self.g.get(node, float('inf'))
        rhs = self.rhs.get(node, float('inf'))

        min_val = min(g, rhs)
        key = (min_val + self.cost_heuristic(node), min_val)
        return key

    def update_node(self, node):
        """
        Updates a node for when an edge cost coming into it changes.
        Call this for every node affected by a change in edge weight 
        (for directed graphs, just the nodes on the recieving end of an edge are required.)
        
        Args:
            node (object): node to be updated
        """
        self._update_node(node)



class ARAStar(object):
    def __init__(self, graph, start, goal, cost_heuristic=cost_heuristic_none):
        """
        Sets up basic options for ARA* search
        cost_heuristic is the guiding heuristic (h value) used in ARA*.
        By default, it is the zero function, which will just perform djikstra's algorithm

        Args:
            cost_heuristic (func, optional): function that takes in (current_node, goal_node) and returns a scalar
        """
        self.cost_heuristic = cost_heuristic
        self.graph = graph
        self.start = start
        self.goal = goal
        self.reset()

    def fvalue(self, state):
        g = self.g.get(state, float('inf'))

        return g + self.epsilon * self.cost_heuristic(state)

    def reset(self):
        # Initialize Sets
        self.g = dict()
        self.open_list = OpenList()
        self.closed_list = set()
        self.incons_list = set()

        # Set start and goal costs

        self.g[self.start] = 0.0
        self.g[self.goal] = float('inf')



    def improve_path(self, epsilon):

        while True:
            if self.open_list.empty():
                break

            if self.fvalue(self.goal) <= self.open_list.peek()[0]:
                break

        # while self.fvalue(self.goal) > self.open_list.peek()[0]:

            s = self.open_list.get()
            self.closed_list.add(s)
            for successor, cost in zip(*self.graph.get_successors(s)):
                if self.g.get(successor, float('inf')) > self.g.get(s, float('inf')) + cost:
                    self.g[successor] = self.g.get(s, float('inf')) + cost
                    if successor in self.closed_list:
                        self.incons_list.add(successor)
                    else: # Inserting into Open List
                        if self.open_list.contains(successor):
                            self.open_list.decrease_key(successor, self.fvalue(successor))
                        else:
                            self.open_list.put(successor, self.fvalue(successor))


    def backout_path(self):

        if self.g.get(self.goal, float('inf')) == float('inf'):
            return False, [], float('inf'), 0

        # back out path
        node = self.goal
        path = [self.goal]
        while node != self.start:
            predecessors, _ = self.graph.get_predecessors(node)
            vals = np.array([self.g.get(pred, float('inf')) for pred in predecessors])
            idx = np.argmin(vals)
            node = predecessors[idx]
            path.append(node)
        path.append(self.start)
        path.reverse()

        return True, path, self.g[self.goal], 0



    def search(self, time_s):


        self.reset()
        # node expansions
        nodes_expanded = 0
        self.epsilon = 3

        # Insert start into Open
        self.open_list.put(self.start, self.fvalue(self.start))

        start_time = time.clock()
        self.improve_path(self.epsilon)
        elapsed = time.clock() - start_time

        while self.epsilon > 1 and elapsed < time_s:

            self.epsilon -= .05
            # Update priorities for everything in OPEN

            # TODO: make this less hacky
            # copying over dict of everything in open list
            # and reinserting with new priority
            open_list_dict = copy.deepcopy(self.open_list.heap_dict)
            self.open_list = OpenList()

            for state, _ in open_list_dict.iteritems():
                self.open_list.put(state, self.fvalue(state))

            # Move states from INCOS to Open
            for s in self.incons_list:
                self.open_list.put(s, self.fvalue(s))
            self.incons_list = set()

            self.closed_list = set()
            self.improve_path(self.epsilon)
            elapsed = time.clock() - start_time


        print 'ARA* Time Elapsed: ', elapsed
        return self.backout_path()



class ADStar(object):
    def __init__(self, graph, start, goal, cost_heuristic=cost_heuristic_none):
        """
        Sets up basic options for ARA* search
        cost_heuristic is the guiding heuristic (h value) used in AD*.
        By default, it is the zero function, which will just perform djikstra's algorithm

        Args:
            cost_heuristic (func, optional): function that takes in (current_node, goal_node) and returns a scalar
        """

        self.cost_heuristic = cost_heuristic
        self.graph = graph
        self.start = start
        self.goal = goal
        self.reset()

    def reset(self):
        # Initialize Sets
        self.open_list = OpenList()
        self.closed_list = set()
        self.incons_list = set()

        # Initialize Costs and Back pointers
        self.bp = dict()
        self.g = dict()
        self.v = dict()

        # set initial g,v values
        self.g[self.start] = 0
        self.g[self.goal] = float('inf')

        self.v[self.start] = float('inf')
        self.v[self.goal] = float('inf')
        self.bp[self.start] = None
        self.bp[self.goal] = None
        self.epsilon = 1

        # add start node to open list
        h = self.cost_heuristic(self.start)
        self.open_list.put(self.start, (h, 0))




    def key(self, state):
        v = self.v.get(state,float('inf'))
        g = self.g.get(state,float('inf'))
        if (v >= g):
            return g + self.epsilon * self.cost_heuristic(state), g
        else:
            return v + self.cost_heuristic(state), v

    def update_graph(self, graph):
        self.graph = graph

    def update_set_membership(self, state):
        if self.v[state] != self.g[state]:
            if state not in self.closed_list: # If not closed, insert or update key in open
                if self.open_list.contains(state):
                    self.open_list.decrease_key(state, self.key(state))
                else:
                    self.open_list.put(state, self.key(state))
            elif state not in self.incons_list:
                self.incons_list.add(state)
        else:
            if self.open_list.contains(state):
                self.open_list.remove(state) # lolz
            elif state in self.incons_list:
                self.incons_list.remove(state)

    def compute_path(self, epsilon):

        self.epsilon = epsilon
        while True:
            if self.open_list.empty():
                break # No path exists

            if self.key(self.goal) <= self.open_list.peek()[0] and (self.v[self.goal] >= self.g[self.goal]):
                 break

            s = self.open_list.get() # Min element from Open

            # IF overconsistent
            if self.v[s] > self.g[s]:
                self.v[s] = self.g[s]
                # Add to closed list
                self.closed_list.add(s)
                for successor, cost in zip(*self.graph.get_successors(s)):
                    # IF have never seen this
                    if not self.g.has_key(successor):
                        self.v[successor] = self.g[successor] = float('inf')
                        self.bp[successor] = None
                    if self.g[successor] > self.g[s] + cost:
                        self.bp[successor] = s
                        self.g[successor] = self.g[s] + cost
                        self.update_set_membership(successor)
            else: # Propagating underconsistency
                print 'derpy derpy derp'
                self.v[s] = float('inf')
                self.update_set_membership(s)
                for successor, cost in zip(*self.graph.get_successors(s)):
                    if not self.g.has_key(successor):
                        self.v[successor] = self.g[successor] = float('inf')
                        self.bp[successor] = None
                    if self.bp[successor] == s:
                        self.update_best_pred(successor)
                        self.update_set_membership(successor)



        path = []
        curr_node = self.goal
        while (curr_node is not None):
            path.append(curr_node)
            curr_node = self.bp[curr_node]

        path.reverse()
        nodes_expanded = 0
        return True, path, self.g[self.goal], nodes_expanded



    # Call update graph before calling this
    def update_node(self, node):
        if node != self.start and self.g.has_key(node):
            self.update_best_pred(node); self.update_set_membership(node)


    def update_best_pred(self, node):
        # Check all predecessors of successor::
        min_pred = None
        min_cost = float('inf')
        for pred, cost in zip(*self.graph.get_predecessors(node)):
            g_val = self.v[pred] + cost
            if g_val < min_cost:
                min_pred = pred
                min_cost = g_val
        self.bp[node] = min_pred
        self.g[node] = min_cost


if __name__ == '__main__':
    q = OpenList()
    q.put('a', 1)
    q.put('b', -1)
    q.put('c', 2)
    q.put('d', 0)

    print(q.heap)
    q.decrease_key('c', -2)
    print(q.heap)


