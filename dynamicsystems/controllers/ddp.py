import numpy as np
import scipy.linalg
import math
import sys
sys.path.append('../')
from ode import ode_solver_once


def numerical_quadraticization(func, x0, u0):
    dx = 0.01
    du = 0.01
    f0 = func(x0, u0)

    n = len(x0)
    m = len(u0)
    lx = np.zeros((n, 1))
    for i in range(n):
        vec_dx = np.zeros((n, 1))
        vec_dx[i] = dx
        new_f_x = func(x0 + vec_dx, u0) 
        delta_f_x = (new_f_x - f0) / dx
        lx[i] = delta_f_x

    lu = np.zeros((m, 1))
    for i in range(m):
        vec_du = np.zeros((m, 1))
        vec_du[i] = du
        new_f_u = func(x0 + vec_du, u0) 
        delta_f_u = (new_f_u - f0) / du
        lu[i] = delta_f_u


    lxx = np.zeros((n, n))
    for i in range(n):
        


# differential dynamic programming
class DDP(object):
    def __init__(self, sys, cost_step_func):
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

    def _backward_pass(self, x_list, u_list, dt, cost_step_func):
        linearized_list = []
        for x, u in zip(x_list, u_list):
            f0, A, B = self.sys.get_linearization(x, u)

            # discretize our linearized system
            f_x = np.eye(A.shape[0]) + A * dt
            f_u = B * dt

            # quadraticize costs

            # quadraticize cost_step_func
            l_x = 



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


        