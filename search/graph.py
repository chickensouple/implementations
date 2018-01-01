import numpy as np
import scipy
import skimage.measure
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import math

def _generate_rooms(size, nrooms, max_room_size):
    """
    Generates a random 
    map with rooms and connecting hallways
    :param size: (nrows, ncols)
    :param nrooms: number of rooms to generate
    :param max_room_size: maximum size of a room
    :return: numpy array representing the map
    1 is traversable region, 0 is not
    """
    arr = np.zeros((size, size), dtype=np.int8)

    for i in range(nrooms):
        rand_row_start = np.random.randint(size)
        rand_col_start = np.random.randint(size)

        rand_row_size = np.random.randint(max_room_size / 2, max_room_size)
        rand_col_size = np.random.randint(max_room_size / 2, max_room_size)

        arr[rand_row_start:rand_row_start + rand_row_size, rand_col_start:rand_col_start + rand_col_size] = 1

    labels = skimage.measure.label(arr)
    regions = skimage.measure.regionprops(labels)

    centroids = list()
    for region in regions:
        centroids.append(region.centroid)

    num_centroids = len(centroids)

    # get distances between every pair of centroids
    dists = scipy.spatial.distance.cdist(centroids, centroids)

    # get a distance that is greater than all current distances
    max_dist = np.max(dists) + 1

    # make sure upper triangle is at least max_dist so that when picking closest
    # pairs, we won't choose a diagonal element or a duplicate connection
    dists = dists + np.triu(np.ones((num_centroids, num_centroids))) * max_dist

    for i in range(num_centroids - 1):
        min_dist_idx = np.argmin(dists)
        min_dist_idx = np.unravel_index(min_dist_idx, dists.shape)

        # create a hallway between regionprops
        centroid1 = np.array(centroids[min_dist_idx[0]], dtype=np.int)
        centroid2 = np.array(centroids[min_dist_idx[1]], dtype=np.int)

        [row_centroid_1, row_centroid_2] = sorted([centroid1, centroid2], key=lambda x: x[0])
        [col_centroid_1, col_centroid_2] = sorted([centroid1, centroid2], key=lambda x: x[1])

        arr[row_centroid_1[0]:row_centroid_2[0] + 1, row_centroid_1[1]] = 1
        arr[row_centroid_2[0], col_centroid_1[1]:col_centroid_2[1] + 1] = 1

        dists[:, min_dist_idx[1]] += max_dist

    return arr


class GraphBase(object):
    def get_successors(self, node):
        raise Exception('Not Implemented')

    def get_predecessors(self, node):
        raise Exception('Not Implemented')

    def get_cost(self, node1, node2):
        raise Exception('Not Implemented')

class MapGraph(GraphBase):
    def __init__(self, maptype='empty', size=20, connectivity=4):
        super(MapGraph, self).__init__()
        # ones in the array are traversable
        # zeros are not
        if maptype == 'empty':
            self.arr = np.ones((size, size), dtype=np.int8)
        elif maptype == 'rooms':
            nrooms = int(size / 5)
            self.arr = _generate_rooms(size, nrooms, int(math.pow(size/1.2, 0.8)))
        else:
            raise Exception('Invalid maptype')

        if connectivity != 4 and connectivity != 8:
            raise Exception('Connectivity can only be 4 or 8 way')
        self.connectivity = connectivity

    def get_successors(self, node):
        # if the node is occupied, there are no successors
        if self.arr[node[0], node[1]]==0:
            return [], []

        shape = self.arr.shape
        
        neighbors = []
        costs = []

        # negative x direction is available for expansion
        x_neg = (node[0] > 0)
        # positive x direction is available for expansion
        x_pos = (node[0] < shape[0]-1)
        # negative y direction is available for expansion
        y_neg = (node[1] > 0)
        # positive y direction is available for expansion
        y_pos = (node[1] < shape[1]-1)

        if x_neg and (self.arr[node[0]-1, node[1]]==1):
            neighbors.append((node[0]-1, node[1]))
            costs.append(1.)
        if x_pos and (self.arr[node[0]+1, node[1]]==1):
            neighbors.append((node[0]+1, node[1]))
            costs.append(1.)
        if y_neg and (self.arr[node[0], node[1]-1]==1):
            neighbors.append((node[0], node[1]-1))
            costs.append(1.)
        if y_pos and (self.arr[node[0], node[1]+1]==1):
            neighbors.append((node[0], node[1]+1))
            costs.append(1.)

        if self.connectivity == 4:
            return neighbors, costs

        # diagonal neighbors
        sqrt_two = math.sqrt(2.)
        if x_neg and y_neg and (self.arr[node[0]-1, node[1]-1]==1):
            neighbors.append((node[0]-1, node[1]-1))
            costs.append(sqrt_two)
        if x_neg and y_pos and (self.arr[node[0]-1, node[1]+1]==1):
            neighbors.append((node[0]-1, node[1]+1))
            costs.append(sqrt_two)
        if x_pos and y_neg and (self.arr[node[0]+1, node[1]-1]==1):
            neighbors.append((node[0]+1, node[1]-1))
            costs.append(sqrt_two)
        if x_pos and y_pos and (self.arr[node[0]+1, node[1]+1]==1):
            neighbors.append((node[0]+1, node[1]+1))
            costs.append(sqrt_two)

        return neighbors, costs

    def get_predecessors(self, node):
        return self.get_successors(node)

    def plot(self, path=None, ax=None):
        if ax == None:
            ax = plt.gca()

        ax.cla()
        ax.imshow(self.arr.T, interpolation='none', vmin=0, vmax=1, origin='lower', cmap='gray')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        if path != None:
            for i in range(1, len(path)):
                node1 = path[i-1]
                node2 = path[i]
                ax.plot([node1[0], node2[0]], [node1[1], node2[1]], c='r')

            path_arr = np.array(path)
            ax.scatter(path_arr[:, 0], path_arr[:, 1], c='r')
            ax.scatter(path_arr[0, 0], path_arr[0, 1], c='g')
            ax.scatter(path_arr[-1, 0], path_arr[-1, 1], c='b')


class MapGraphCar(MapGraph):
    def __init__(self, cartype='dubins', **kwargs):
        super(MapGraphCar, self).__init__(**kwargs)

        if cartype != 'dubins' and cartype != 'reed-shepp':
            raise Exception('Cartype must be "dubins" or "reed-shepp"')
        self.cartype = cartype

        # state consist of (pos_x, pos_y, dir_x, dir_y)
        # where dir=(1, 0) is positive x, 
        #       dir=(0, 1) is positive y, 
        #       dir=(-1, 0) is negative x,
        #       dir=(0, -1) is negative y

    def _valid_pos(self, pos):
        shape = self.arr.shape
        valid = pos[0] >= 0 and pos[0] < shape[0] and \
                pos[1] >= 0 and pos[1] < shape[1] and \
                self.arr[pos[0], pos[1]]==1
        return valid

    def get_successors(self, state):
        if self.arr[state[0], state[1]]==0:
            return [], []

        forward_pos = (state[0]+state[2], state[1]+state[3], state[2], state[3])
        left_pos = (forward_pos[0]-state[3], forward_pos[1]+state[2], -state[3], state[2])
        right_pos = (forward_pos[0]+state[3], forward_pos[1]-state[2], state[3], -state[2])

        neighbors = []
        costs = []

        pi_over_2 = math.pi * 0.5
        # if forward position is valid
        if self._valid_pos(forward_pos):
            neighbors.append(forward_pos)
            costs.append(1.)

            # turns available only when forward pos is available 
            # since it passes through the forward grid position

            # if left turn is available 
            if self._valid_pos(left_pos):
                neighbors.append(left_pos)
                costs.append(pi_over_2)

            # if right turn is available
            if self._valid_pos(right_pos):
                neighbors.append(right_pos)
                costs.append(pi_over_2)

        if self.cartype == 'dubins':
            return neighbors, costs

        # add backward neighbors for reed-shepp
        neg_pos = (state[0]-state[2], state[1]-state[3], state[2], state[3])
        neg_left_pos = (neg_pos[0]-state[3], neg_pos[1]+state[2], state[3], -state[2])
        neg_right_pos = (neg_pos[0]+state[3], neg_pos[1]-state[2], -state[3], state[2])

        # if backward position is valid
        if self._valid_pos(neg_pos):
            neighbors.append(neg_pos)
            costs.append(1.)

            # turns available only when forward pos is available 
            # since it passes through the forward grid position

            # if left turn is available 
            if self._valid_pos(neg_left_pos):
                neighbors.append(neg_left_pos)
                costs.append(pi_over_2)

            # if right turn is available
            if self._valid_pos(neg_right_pos):
                neighbors.append(neg_right_pos)
                costs.append(pi_over_2)

        return neighbors, costs


    def get_predecessors(self, state):
        if self.cartype == 'dubins':
            raise Exception('Not Yet Implemented')
        elif self.cartype == 'reed-shepp':
            return self.get_successors(state)

    def plot(self, path=None, ax=None):
        if ax == None:
            ax = plt.gca()

        ax.cla()
        ax.imshow(self.arr.T, interpolation='none', vmin=0, vmax=1, origin='lower', cmap='gray')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        def draw_curve(ax, start, end):
            forward_pos = (start[0]+start[2], start[1]+start[3])
            left_pos = (forward_pos[0]-start[3], forward_pos[1]+start[2])
            right_pos = (forward_pos[0]+start[3], forward_pos[1]-start[2])
            neg_pos = (start[0]-start[2], start[1]-start[3])
            neg_left_pos = (neg_pos[0]-start[3], neg_pos[1]+start[2])
            neg_right_pos = (neg_pos[0]+start[3], neg_pos[1]-start[2])

            
            end_pos = end[:2]

            if (end_pos == forward_pos or end_pos == neg_pos):
                # forward
                ax.plot([start[0], end[0]], [start[1], end[1]], c='r')
            elif (end_pos == left_pos):
                # left
                center = (start[0]-start[3], start[1]+start[2])
                delta = (start[0]-center[0], start[1]-center[1])

                start_theta = math.atan2(delta[1], delta[0]) * 180. / math.pi
                arc = Arc(center, 2, 2, theta1=start_theta, theta2=start_theta+90, color='r')
                ax.add_patch(arc)
            elif (end_pos == right_pos):
                # right
                center = (start[0]+start[3], start[1]-start[2])
                delta = (start[0]-center[0], start[1]-center[1])

                start_theta = math.atan2(delta[1], delta[0]) * 180. / math.pi
                arc = Arc(center, 2, 2, theta2=start_theta, theta1=start_theta-90, color='r')
                ax.add_patch(arc)
            elif (end_pos == neg_left_pos):
                draw_curve(ax, end, start)
            elif (end_pos == neg_right_pos):
                draw_curve(ax, end, start)

        if path != None:
            for i in range(1, len(path)):
                node1 = path[i-1]
                node2 = path[i]

                # swapping x and y to draw on image
                # node1 = (node1[1], node1[0], node1[3], node1[2])
                # node2 = (node2[1], node2[0], node2[3], node2[2])
                draw_curve(ax, node1, node2)

            path_arr = np.array(path)
            ax.scatter(path_arr[:, 0], path_arr[:, 1], c='r')
            ax.scatter(path_arr[0, 0], path_arr[0, 1], c='g')
            ax.scatter(path_arr[-1, 0], path_arr[-1, 1], c='b')

            ax.arrow(path_arr[0, 0], path_arr[0, 1], path_arr[0, 2], path_arr[0, 3], 
                head_width=0.4, head_length=0.3, color='g')
            ax.arrow(path_arr[-1, 0], path_arr[-1, 1], path_arr[-1, 2], path_arr[-1, 3], 
                head_width=0.4, head_length=0.3, color='b')


