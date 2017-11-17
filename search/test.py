import queue
from graph import *
from path_planning import *
from heuristics import *

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from functools import partial

    def generate_start_and_goal(mapgraph):
        start = np.random.randint(m.arr.shape[0], size=(2))
        while mapgraph.arr[start[0], start[1]] == 0:
            start = np.random.randint(m.arr.shape[0], size=(2))

        goal = np.random.randint(m.arr.shape[0], size=(2))
        while mapgraph.arr[goal[0], goal[1]] == 0 or np.all(start == goal):
            goal = np.random.randint(m.arr.shape[0], size=(2))

        start = tuple(start)
        goal = tuple(goal)
        return start, goal

    seed = np.random.randint(2**31)
    # seed = 385233812
    np.random.seed(seed)
    print("Seed: " + str(seed))
    m = MapGraph(size=80, maptype='rooms')
    start, goal = generate_start_and_goal(m)
    print("Start: " + str(start))
    print("Goal: " + str(goal))

    heuristic = partial(map_heuristic_l1, goal=goal)
    # path_found, path, cost = dijkstra(m, start, goal)
    path_found, path, cost = astar(m, start, goal, heuristic)

    if path_found:
        print("Path Found")
        print("Cost: " + str(cost))
        print(path)

        m.plot(path)
        plt.show()
    else:
        print("Path Not Found")

        m.plot()
        plt.show()

