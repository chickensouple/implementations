import numpy as np
import scipy
import skimage.measure
import matplotlib.pyplot as plt
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
    def get_neighbors(self, node):
        raise Exception('Not Implemented')

    def get_cost(self, node1, node2):
        raise Exception('Not Implemented')

class MapGraph(GraphBase):
    def __init__(self, maptype='empty', size=20):
        super(MapGraph, self).__init__()
        # ones in the array are traversable
        # zeros are not
        if maptype == 'empty':
            self.arr = np.ones((size, size), dtype=np.int8)
        elif maptype == 'rooms':
            nrooms = int(size / 5)
            self.arr = _generate_rooms(size, nrooms, int(math.pow(size/2, 0.8)))
        else:
            raise Exception('Invalid maptype')

    def get_neighbors(self, node):
        shape = self.arr.shape
        
        neighbors = []
        if (node[0] > 0) and (self.arr[node[0]-1, node[1]]==1):
            neighbors.append((node[0]-1, node[1]))
        if (node[0] < shape[0]-1) and (self.arr[node[0]+1, node[1]]==1):
            neighbors.append((node[0]+1, node[1]))
        
        if (node[1] > 0) and (self.arr[node[0], node[1]-1]==1):
            neighbors.append((node[0], node[1]-1))
        if (node[1] < shape[1]-1) and (self.arr[node[0], node[1]+1]==1):
            neighbors.append((node[0], node[1]+1))
        
        return neighbors

    def get_cost(self, node1, node2):
        # TODO: check if nodes are neighbors
        return 1

    def plot(self, path=None):
        plt.cla()
        plt.imshow(self.arr)

        if path != None:
            path_arr = np.array(path)
            plt.scatter(path_arr[:, 1], path_arr[:, 0], c='r')
            plt.scatter(path_arr[0, 1], path_arr[0, 0], c='g')
            plt.scatter(path_arr[-1, 1], path_arr[-1, 0], c='b')







