from graph import *
from graph_search import *
from heuristics import *

import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import sys
import threading
import copy
import signal
from functools import partial

try:
    # python 2
    import Tkinter as tk
except:
    # python 3
    import tkinter as tk


class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


class MapData(object):
    def __init__(self):
        self.data_lock = threading.Lock()

        self.graph = MapGraph(size=30, maptype='empty', connectivity=8)
        # self.graph = MapGraphCar(cartype='reed-shepp', size=30, maptype='empty', connectivity=8)

        self.start = None
        self.goal = None

        self.graph_changed = False
        self.start_goal_changed = False

        self.changed_node_set = set()

    def get_graph(self):
        self.data_lock.acquire()
        graph = copy.deepcopy(self.graph)
        self.data_lock.release()
        return graph

    def set_graph_loc(self, loc, val):
        self.data_lock.acquire()
        if val != self.graph.arr[loc]:
            for succ in self.graph.get_successors(loc)[0]:
                self.changed_node_set.add(succ)
            self.graph.arr[loc] = val
            self.changed_node_set.add(loc)

        self.graph_changed = True
        self.data_lock.release()

    def get_graph_loc(self, loc):
        self.data_lock.acquire()
        val = copy.deepcopy(self.graph.arr[loc])
        self.data_lock.release()
        return val

    def set_start(self, start):
        self.data_lock.acquire()
        self.start = start
        self.start_goal_changed = True
        self.data_lock.release()

    def set_goal(self, goal):
        self.data_lock.acquire()
        self.goal = goal
        self.start_goal_changed = True
        self.data_lock.release()

    def get_start_goal(self):
        self.data_lock.acquire()
        start = copy.deepcopy(self.start)
        goal = copy.deepcopy(self.goal)
        self.data_lock.release()
        return start, goal

    def get_flags(self):
        self.data_lock.acquire()
        graph_changed = self.graph_changed
        start_goal_changed = self.start_goal_changed
        self.data_lock.release()
        return graph_changed, start_goal_changed

    def clear_flags(self):
        self.data_lock.acquire()
        self.graph_changed = False
        self.start_goal_changed = False
        self.data_lock.release()

    def clear_changed_data(self):
        self.data_lock.acquire()
        self.changed_node_set = set()
        self.data_lock.release()

    def get_changed_data(self):
        self.data_lock.acquire()
        changed_node_set = copy.deepcopy(self.changed_node_set)
        self.data_lock.release()
        return changed_node_set

    def draw_map(self, ax, path=None):
        # TODO: this locking time can be made much shorter by copying array out
        self.data_lock.acquire()
        if path == None:
            self.graph.plot(ax=ax)
        else:
            self.graph.plot(ax=ax, path=path)
        self.data_lock.release()

        start, goal = self.get_start_goal()
        if start != None:
            ax.scatter(start[0], start[1], c='g')

        if goal != None:
            ax.scatter(goal[0], goal[1], c='b')


class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.map_data = MapData()

        self.adstar_thread = None
        self.adstar = None

        self.found_path_lock = threading.Lock()
        self.found_path = False

        self.path_changed_lock = threading.Lock()
        self.path_changed = False  # for triggering redraw after path is found

        self.create_widgets()
        self.graceful_killer = GracefulKiller()

        self.update()

    def on_click_event(self, event):
        try:
            x, y = event.inaxes.transData.inverted().transform((event.x, event.y))
        except:
            return
        x = int(round(x))
        y = int(round(y))
        loc = (x, y)

        if event.button == 1:
            # left click
            self.map_data.set_graph_loc(loc, 0)
        elif event.button == 3:
            # right click
            self.map_data.set_graph_loc(loc, 1)

    def on_key_event(self, event):
        try:
            x, y = event.inaxes.transData.inverted().transform((event.x, event.y))
        except:
            return
        x = int(round(x))
        y = int(round(y))

        loc = (x, y)

        if self.map_data.get_graph_loc(loc) == 0:
            print("Not Valid Location")
            return

        if event.key == 's':
            self.map_data.set_start(loc)
            # self.map_data.set_start((x, y, 0, 1))
        elif event.key == 'g':
            self.map_data.set_goal(loc)
            # self.map_data.set_goal((x, y, 0, 1))

    def create_widgets(self):
        fig = plt.figure(figsize=(8, 8))
        self.ax = fig.add_subplot(111)

        self.map_data.draw_map(ax=self.ax)

        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect('button_press_event', self.on_click_event)
        self.canvas.mpl_connect('key_press_event', self.on_key_event)
        self.canvas.show()

    def draw_map(self):
        self.found_path_lock.acquire()
        found_path = self.found_path
        if self.found_path:
            path = copy.deepcopy(self.path)
        else:
            path = None
        self.found_path_lock.release()

        self.map_data.draw_map(self.ax, path)
        self.canvas.draw()

    def run_adstar(self):
        path_found, path, cost, nodes_expanded = self.adstar.compute_path(1.0)
        self.found_path_lock.acquire()
        self.found_path = path_found
        self.path = copy.deepcopy(path)
        self.found_path_lock.release()

        self.cost = cost
        self.nodes_expanded = nodes_expanded
        if path_found:
            print("Path Found")
            print("Cost: " + str(cost))
        else:
            print("Path Not Found")
        print("Nodes expanded: " + str(nodes_expanded))

        # so that graph will be redrawn
        self.path_changed_lock.acquire()
        self.path_changed = True
        self.path_changed_lock.release()

    def update(self):
        if self.graceful_killer.kill_now:
            exit()

        self.path_changed_lock.acquire()
        path_changed = self.path_changed
        self.path_changed_lock.release()

        graph_changed, start_goal_changed = self.map_data.get_flags()
        start, goal = self.map_data.get_start_goal()

        if graph_changed or start_goal_changed or path_changed:
            self.draw_map()

            # clear flag to draw path
            self.path_changed_lock.acquire()
            self.path_changed = False
            self.path_changed_lock.release()

        if (start_goal_changed or graph_changed) and start != None and goal != None:
            # only start new adstar thread if there isn't one or its finished
            if self.adstar_thread == None or not self.adstar_thread.isAlive():

                heuristic = partial(cost_heuristic_linf, goal=goal)
                graph = self.map_data.get_graph()

                if start_goal_changed:
                    # only create new ADStar star if start or goal is changed
                    self.adstar = ADStar(graph, start, goal, cost_heuristic=heuristic)
                elif graph_changed:
                    # update adstar with changed nodes
                    self.adstar.update_graph(graph)
                    change_node_set = self.map_data.get_changed_data()
                    for node in change_node_set:
                        self.adstar.update_node(node)

                self.map_data.clear_changed_data()

                self.found_path_lock.acquire()
                self.found_path = False
                self.found_path_lock.release()

                self.adstar_thread = threading.Thread(target=self.run_adstar)
                self.adstar_thread.run()

                self.map_data.clear_flags()

        # refresh every 0.05 seconds
        self.after(int(0.05 * 1e3), self.update)


if __name__ == '__main__':
    def _delete_window():
        exit()


    def _destroy(event):
        exit()


    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", _delete_window)
    root.bind("<Destroy>", _destroy)

    app = Application(master=root)
    app.mainloop()
