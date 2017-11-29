import numpy as np
import scipy.linalg

# differential dynamic programming
class DDP(object):
    def __init__(self, sys, cost_func):
        self.sys = sys
        self.cost_func = cost_func

    def solve(self, x0, T):
        U = None
        self._forward_pass(x0, U)



    def _forward_pass(self, x0, U):
        x_list = [x0]
        self.sys.set_state(x0)
        for control in U:
            x_list.append(self.sys.step(control))

    def _backward_pass(self):
        pass



        