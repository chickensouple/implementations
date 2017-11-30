import numpy as np
import scipy.linalg
import math
import sys
sys.path.append('../')
from ode import ode_solver_once

# differential dynamic programming
class DDP(object):
    def __init__(self, sys, cost_func):
        self.sys = sys
        self.cost_func = cost_func

    def solve(self, x0, N, dt):
        u_list = [np.array([[0]]) for _ in range(N)]
        x_list = [x0]
        for i, control in enumerate(u_list):
            #f, x, u, dt
            x_new = ode_solver_once(self.sys.get_diff_eq(), x_list[-1], u_list[i], dt)
            x_list.append(x_new)

        self._backward_pass(x_list, u_list, dt)

    def _forward_pass(self, x0, U):
        x_list = [x0]
        self.sys.set_state(x0)
        for control in U:
            x_list.append(self.sys.step(control))

    def _backward_pass(self, x_list, u_list, dt):
        linearized_list = []
        for x, u in zip(x_list, u_list):
            f0, A, B = self.sys.get_linearization(x, u)

            # turn affine system into linear system
            A_lin = np.zeros((A.shape[0]+1, A.shape[1]+1))
            A_lin[:A.shape[0], :A.shape[1]] = A
            A_lin[:-1, -1] = f0.squeeze()
            A_lin[-1, -1] = 1

            B_lin = np.zeros((B.shape[0]+1, B.shape[1]))
            B_lin[:-1, :] = B


            # discretize our continuous affine system
            A_disc = np.eye(A_lin.shape[0]) + A_lin * dt
            B_disc = B_lin * dt



            # quadraticize costs


            linearized_list.append((A_disc, B_disc))




if __name__ == '__main__':
    import models


    def final_cost(x, target):
        cost = np.linalg.norm(x - target)
        # cost = np.sum(np.square(x - target))
        return cost


    def step_cost(x, u, target):
        cost = np.linalg.norm(x - target)
        cost += np.linalg.norm(u)
        return cost    


    pendulum = models.Pendulum()
    ddp = DDP(pendulum, None)

    x0 = np.array([[math.pi, 0.]]).T
    N = 10
    dt = 0.05

    ddp.solve(x0, N, dt)


        