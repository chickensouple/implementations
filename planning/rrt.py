from tree import Tree
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')

class RRT(object):
    def __init__(self, config):
        """
        Initializes RRT
        
        Args:
            config (dict): dict must contain following keys and values
            'collision_check', function that takes in node and map and returns true of collide
            'random_sample', function that generates a random_sample
            'steer', returns a path that goes from one node towards another
            'dist', a measure of distance used for nearest neighbor search
            'goal_region', function that returns true if node is in goal region
        """
        self.config = config
        self.clear()

    def clear(self):
        self.tree = Tree()
        self.found_path = False


    def find_path(self, map_info, maxnodes=5000, show=False):
        self.clear()
        self.goal_node_idx = None

        self.tree.insert_node(map_info['start'], [map_info['start']])

        i = 0
        while (i < maxnodes):
            # print env
            rand_node = self.config['random_sample'](map_info)
            closest_idx = self.tree.closest_idx(rand_node, self.config['dist'])
            closest_node = self.tree.node_states[closest_idx]

            path, path_cost = self.config['steer'](closest_node, rand_node)
            new_node = path[-1]

            if self.config['collision_check'](map_info['map'], path):
                continue

            self.tree.insert_node(new_node, path, closest_idx)

            if self.config['goal_region'](new_node, map_info['goal']):
                self.found_path = True
                self.goal_node_idx = len(self.tree.node_states)-1
                break

            if show and i % 20 == 0:
                plt.cla()
                self.tree.show(goal=map_info['goal'])
                plt.show(block=False)
                plt.pause(0.01)

            i += 1

    def show(self, map_info):
        plt.cla()
        if self.found_path:
            self.tree.show(goal=map_info['goal'], path_idx=len(self.tree.node_states)-1)
        else:
            self.tree.show(goal=map_info['goal'])


if __name__ == '__main__':
    import models
    from functools import partial
    import math
    pendulum = models.Pendulum()



    class PendulumFuncs(object):
        def __init__(self):
            self.pendulum = models.Pendulum()

        def get_config(self):
            config = {'collision_check': partial(PendulumFuncs.collision_check, self),
                      'random_sample': partial(PendulumFuncs.random_sample, self),
                      'steer': partial(PendulumFuncs.pendulum_steer, self),
                      'dist': partial(PendulumFuncs.dist, self),
                      'goal_region': partial(PendulumFuncs.goal_region, self)}
            return config

        def collision_check(self, env, path):
            return False

        def random_sample(self, env):
            rand_theta = np.random.random_sample() * 2 * np.pi

            magnitude = 7
            rand_theta_dot = np.random.random() * (2 * magnitude) - magnitude

            return np.array([rand_theta, rand_theta_dot])

        def dist(self, nodes_from, node_to):
            dist = nodes_from - node_to
            dist = np.square(dist)
            dist = np.sum(dist, axis=1)
            dist = np.sqrt(dist)
            return dist

        def pendulum_steer(self, closest_node, target_node):
            dt = 0.1
            max_steps = 10

            min_dist = np.Inf
            best_u = None
            reached_x = None

            for i in range(15):
                u = np.array([np.random.random() * 2 - 1])
                control_func = models.ode.constant_controller(u)
                
                num_steps = np.random.randint(1, max_steps)

                dts = np.ones(num_steps) * dt
                x1, t1 = models.ode.ode_solver(self.pendulum.get_diff_eq(), control_func, np.array([closest_node]).T, 0, dts)
                
                x_end = x1[:, -1]
                dist = self.dist(np.array([x_end]), target_node)
                
                if dist < min_dist:
                    min_dist = dist
                    best_u = u
                    reached_x = x_end

            return [reached_x], 1


        def goal_region(self, node, goal):
            if (abs(node[0] - goal[0]) < 0.1) and \
                (abs(node[1] - goal[1]) < 0.1):
                return True
            else:
                return False


    p = PendulumFuncs()
    config = p.get_config()


    data_dict = {
        'map': None,
        'start': np.array([math.pi, 0]),
        'goal': np.array([0, 0])
    }

    rrt = RRT(config)
    rrt.find_path(data_dict, show=False)

    rrt.show(data_dict)
    plt.show()



    # p.pendulum_steer(np.array([0, 1]), np.array([1, 1]))

    # import matplotlib.pyplot as plt
    # from generate_data import generate_data
    # from functools import partial
    # from rrt_utils import *
    # from map_utils import *

    # # np.random.seed(20)
    # data_dict = generate_data('rooms', dubins=False)
    # # np.random.seed(int(round(time.time() * 1000)) % 2**32)
    # random_sampler = partial(map_sampler_goal_bias, goal=data_dict['goal'], eps=0.1, dubins=False)
    # l2_goal = partial(l2_goal_region, goal=data_dict['goal'])

    # config = {'collision_check': map_collision_check,
    #           'random_sample': random_sampler,
    #           'steer': holonomic_steer,
    #           'dist': l2_dist,
    #           'goal_region': l2_goal}




    # rrt = RRT(config)
    # rrt.find_path(data_dict['map'], data_dict['start'], data_dict['goal'], show=True)
    
    # rrt.tree.show(im=data_dict['map'], goal=data_dict['goal'], path_idx=len(rrt.tree.node_states)-1)
    # plt.show()
