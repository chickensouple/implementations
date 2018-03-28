from graph import *
from graph_search import *
from heuristics import *

def check_valid_pos(mapgraph, pos):
    if (pos[0] < 0 or pos[0] >= mapgraph.arr.shape[0]-1):
        return False
    if (mapgraph.arr[pos[0], pos[1]] == 0):
        return False
    return True

def generate_start_and_goal_grid(mapgraph):
    start = np.random.randint(mapgraph.arr.shape[0], size=(2))
    while not check_valid_pos(mapgraph, start):
        start = np.random.randint(mapgraph.arr.shape[0], size=(2))


    goal = np.random.randint(mapgraph.arr.shape[0], size=(2))
    while not check_valid_pos(mapgraph, goal) or np.all(start == goal):
        goal = np.random.randint(mapgraph.arr.shape[0], size=(2))

    start = tuple(start)
    goal = tuple(goal)
    return start, goal

def generate_start_and_goal_car(mapgraph):
    def gen_direction():
        direction = np.random.randint(4)
        if (direction == 0):
            return np.array([1, 0])
        if (direction == 1):
            return np.array([-1, 0])
        if (direction == 2):
            return np.array([0, 1])
        if (direction == 3):
            return np.array([0, -1])

    start = np.random.randint(mapgraph.arr.shape[0], size=(2))
    start_dir = gen_direction()
    while not check_valid_pos(mapgraph, start):
        start = np.random.randint(mapgraph.arr.shape[0], size=(2))
        start_dir = gen_direction()

    goal = np.random.randint(mapgraph.arr.shape[0], size=(2))
    goal_dir = gen_direction()
    while not check_valid_pos(mapgraph, goal) or np.all(start == goal):
        goal = np.random.randint(mapgraph.arr.shape[0], size=(2))
    goal_dir = gen_direction()

    start = tuple(np.append(start, start_dir))
    goal = tuple(np.append(goal, goal_dir))
    return start, goal

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from functools import partial

    prob_type = 'grid' # ['grid', 'car']

    seed = np.random.randint(2**31)
    seed = 684468469
    # seed = 2010841075 # TODO: car seed bad
    np.random.seed(seed)
    print("Seed: " + str(seed))

    if prob_type == 'grid':
        m = MapGraph(size=80, maptype='rooms', connectivity=8)
        start, goal = generate_start_and_goal_grid(m)
        heuristic = partial(cost_heuristic_l2, goal=goal)
    elif prob_type == 'car':
        m = MapGraphCar(cartype='reed-shepp', size=30, maptype='rooms')
        start, goal = generate_start_and_goal_car(m)
        heuristic = lambda node: cost_heuristic_l2(node[0:2], goal[0:2])
        # heuristic = cost_heuristic_none
    elif prob_type == 'aratest':
        m = MapGraph(maptype='aratest', connectivity=8)
        start = (0, 0)
        goal = (6, 5)
        heuristic = partial(cost_heuristic_linf, goal=goal)
    elif prob_type == 'cartest':
        m = MapGraphCar(cartype='dubins', maptype='cartest')
        start = (0, 16, 0, -1)
        goal = (0, 16, 0, 1)
        heuristic = partial(cost_heuristic_linf, goal=goal)


    # path_found, path, cost, nodes_expanded = astar(m, start, goal, heuristic, tie_heuristic=tie_heuristic_high_g)
    #lpastar = LPAStar(m, start, goal, cost_heuristic=heuristic)
    #path_found, path, cost, nodes_expanded = lpastar.search()
    arastar = ARAStar(m, start, goal, cost_heuristic=heuristic)
    path_found, path, cost, nodes_expanded = arastar.search(1)


    print("Nodes expanded: " + str(nodes_expanded))
    if path_found:
        print("Path Found")
        print("Cost: " + str(cost))
        print(path)

        m.plot(path)
        plt.show()
    else:
        print("Path Not Found")

        m.plot()
        plt.scatter(start[0], start[1], c='g')
        plt.scatter(goal[0], goal[1], c='b')
        plt.show()


