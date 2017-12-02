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
        import pdb
        pdb.set_trace()
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
        for j in range(i+1):
            vec_dx1 = np.zeros((n, 1))
            vec_dx1[i] = dx
            vec_dx2 = np.zeros((n, 1))
            vec_dx2[j] = dx

            f_x1_x2 = func(x0 + vec_dx1 + vec_dx2, u0)
            f_x1 = func(x0 + vec_dx1, u0)
            f_x2 = func(x0 + vec_dx2, u0)

            delta_f_x = (f_x1_x2 - f_x1 - f_x2 + f0) / (dx*dx)

            lxx[i, j] = delta_f_x
            lxx[j, i] = delta_f_x

    luu = np.zeros((m, m))
    for i in range(m):
        for j in range(i+1):
            vec_du1 = np.zeros((m, 1))
            vec_du1[i] = du
            vec_du2 = np.zeros((m, 1))
            vec_du2[j] = du

            f_u1_u2 = func(x0, u0 + vec_du1 + vec_du2)
            f_u1 = func(x0, u0 + vec_du1)
            f_u2 = func(x0, u0 + vec_du2)

            delta_f_u = (f_u1_u2 - f_u1 - f_u2 + f0) / (du*du)

            luu[i, j] = delta_f_u
            luu[j, i] = delta_f_u

    lux = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            vec_du1 = np.zeros((m, 1))
            vec_du1[i] = du
            vec_dx2 = np.zeros((n, 1))
            vec_dx2[j] = dx

            f_u1_x2 = func(x0 + vec_dx2, u0 + vec_du1)
            f_u1 = func(x0, u0 + vec_du1)
            f_x2 = func(x0 + vec_dx2, u0)

            delta_f_ux = (f_u1_x2 - f_u1 - f_x2 + f0) / (du*dx)

            lux[i, j] = delta_f_ux

    # print 'f0:', f0
    # print 'lx:', lx
    # print 'lu:', lu
    # print 'lxx:', lxx
    # print 'luu:', luu
    # print 'lux:', lux

    return f0, lx, lu, lxx, luu, lux




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
        A = None
        B = None
        for x, u in zip(x_list, u_list):
            f0, A, B = self.sys.get_linearization(x, u)

            # discretize our linearized system
            f_x = np.eye(A.shape[0]) + A * dt
            f_u = B * dt



            # quadraticize costs

            # quadraticize cost_step_func
            # l_x = 



if __name__ == '__main__':
    import models
    from functools import partial

    def final_cost(x, target):
        cost = np.linalg.norm(x - target)
        # cost = np.sum(np.square(x - target))
        return cost


    def step_cost(x, u, target):
        cost = np.linalg.norm(x - target)**2
        cost += np.linalg.norm(u)**2
        return cost    


    x0 = np.array([[0, 0]]).T
    u0 = np.array([[0]])
    step_cost_func = partial(step_cost, target=x0)
    numerical_quadraticization(step_cost_func, x0, u0)


    exit()

    pendulum = models.Pendulum()
    ddp = DDP(pendulum, None)

    x0 = np.array([[math.pi, 0.]]).T
    N = 10
    dt = 0.05

    ddp.solve(x0, N, dt)


        