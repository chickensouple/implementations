import numpy as np
import scipy.linalg
import math
import sys
sys.path.append('../')
from ode import ode_solver_once, ode_solver_once_adaptive


def numerical_quadraticization(x0, u0, func):
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

def numerical_quadraticization2(x0, func):
    dx = 0.01
    f0 = func(x0)

    n = len(x0)
    lx = np.zeros((n, 1))
    for i in range(n):
        vec_dx = np.zeros((n, 1))
        vec_dx[i] = dx
        new_f_x = func(x0 + vec_dx) 
        delta_f_x = (new_f_x - f0) / dx
        lx[i] = delta_f_x


    lxx = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            vec_dx1 = np.zeros((n, 1))
            vec_dx1[i] = dx
            vec_dx2 = np.zeros((n, 1))
            vec_dx2[j] = dx

            f_x1_x2 = func(x0 + vec_dx1 + vec_dx2)
            f_x1 = func(x0 + vec_dx1)
            f_x2 = func(x0 + vec_dx2)

            delta_f_x = (f_x1_x2 - f_x1 - f_x2 + f0) / (dx*dx)

            lxx[i, j] = delta_f_x
            lxx[j, i] = delta_f_x

    return f0, lx, lxx



# differential dynamic programming
class DDP(object):
    def __init__(self, sys):
        self.sys = sys
        
    def solve(self, x0, N, dt, quad_step_cost_func, quad_final_cost_func):
        n = len(x0)
        m = 1

        controller_list = [(np.zeros((m, n)), np.zeros((m, 1))) for _ in range(N)]
        x_list = [np.zeros((n, 1)) for _ in range(N)]
        x_list[0] = x0
        u_list = [np.zeros((m, 1)) for _ in range(N)]



        for i in range(50):
            x_list, u_list = self._forward_pass(x_list, u_list, controller_list, dt)
            controller_list = self._backward_pass(x_list, u_list, dt, quad_step_cost_func, quad_final_cost_func)
            # x_arr = np.array(x_list)
            # t = np.ones(N+1) * dt
            # t = np.cumsum(t)
            # plt.cla()
            # plt.plot(t, x_arr[:, 0], label='theta', c='b')
            # plt.plot(t, x_arr[:, 1], label='theta_dot', c='r')
            # plt.legend()
            # plt.show(block=False)
            # plt.pause(0.01)
            # raw_input('Press Enter to Continue: ')

        return controller_list, x_list, u_list

    def _forward_pass(self, x_bar_list, u_bar_list, controller_list, dt):
        x_list = [x_bar_list[0]]
        u_list = []
        for i, (x_bar, u_bar, controller) in enumerate(zip(x_bar_list, u_bar_list, controller_list)):
            x = x_list[-1]
            u = np.dot(controller[0], x - x_bar) + controller[1] + u_bar
            x_new = ode_solver_once(self.sys.get_diff_eq(), x_list[-1], u, dt)

            u_list.append(u)
            x_list.append(x_new)

        return x_list, u_list


    def _backward_pass(self, x_list, u_list, dt, quad_step_cost_func, quad_final_cost_func):
        controller_list = []

        f0, b, A = quad_final_cost_func(x_list[-1])
        for x, u in reversed(zip(x_list, u_list)):
            f0, A_sys, b_sys = self.sys.get_linearization(x, u)

            # discretize our linearized system
            f_x = np.eye(A_sys.shape[0]) + A_sys * dt
            f_u = b_sys * dt

            # quadraticize costs
            f0, lx, lu, lxx, luu, lux = quad_step_cost_func(x, u)

            qx = lx + np.dot(f_x.T, b)
            qu = lu + np.dot(f_u.T, b)
            qxx = lxx + np.dot(f_x.T, np.dot(A, f_x))
            quu = luu + np.dot(f_u.T, np.dot(A, f_u))
            qux = lux + np.dot(f_u.T, np.dot(A, f_x))

            K = -np.linalg.solve(quu, qux)
            j = -np.linalg.solve(quu, qu)
            A = qxx + np.dot(K.T, np.dot(quu, K)) + np.dot(qux.T, K) + np.dot(K.T, qux)
            b = qx + np.dot(K.T, np.dot(quu, j)) + np.dot(qux.T, j) + np.dot(K.T, qu)
            

            controller_list.append((K, j))

        controller_list.reverse()
        return controller_list


if __name__ == '__main__':
    import models
    from functools import partial
    import matplotlib.pyplot as plt


    Cx = np.array([[100., 0], [0, 1]])
    # Cu = np.array([[490.]])
    Cu = np.array([[0.01]])

    def final_cost(x, target_state, Cx=Cx):
        delta_x = x - target_state
        cost = np.dot(delta_x.T, np.dot(Cx, delta_x))
        return cost

    def quad_final_cost(x, target_state, Cx=Cx):
        A = 2*Cx
        b = 2*np.dot(Cx, x - target_state)
        f0 = final_cost(x, target_state)
        return f0, b, A

    def step_cost(x, u, target_state, target_control, Cx=Cx, Cu=Cu):
        delta_x = x - target_state
        delta_u = u - target_control

        cost = np.dot(delta_x.T, np.dot(Cx, delta_x))

        cost += np.dot(delta_u.T, np.dot(Cu, delta_u))
        return cost 

    def quad_step_cost(x, u, target_state, target_control, Cx=Cx, Cu=Cu):
        delta_x = x - target_state
        delta_u = u - target_control

        lx = 2*np.dot(Cx, delta_x)
        lxx = 2*Cx
        lu = 2*np.dot(Cu, delta_u)
        luu = 2*Cu

        m = len(u)
        n = len(x)
        lux = np.zeros((m, n))
        f0 = step_cost(x, u, target_state, target_control)
        return f0, lx, lu, lxx, luu, lux




    target_state = np.array([[0, 0.]]).T
    target_control = np.array([[0.]])
    step_cost_func = partial(step_cost, target_state=target_state, target_control=target_control)
    quad_step_cost_func = partial(quad_step_cost, target_state=target_state, target_control=target_control)

    final_cost_func = partial(final_cost, target_state=target_state)
    quad_final_cost_func = partial(quad_final_cost, target_state=target_state)



    dt = 0.05
    x0 = np.array([[math.pi, 0.]]).T
    pendulum = models.Pendulum(max_torque=np.Inf, dt=dt)
    pendulum.set_state(x0)
    ddp = DDP(pendulum)


    # # open loop controller
    # N = 500
    # controller_list, x_list, u_list = ddp.solve(x0, N, dt, quad_step_cost_func, quad_final_cost_func)
    
    # states = np.zeros((2, N-1))
    # controls = np.zeros(N-1)
    # for i in range(N-1):
    #     state = pendulum.get_state()
    #     states[:, i] = state.squeeze()

    #     controller = controller_list[0]
    #     u = np.dot(controller[0], state - x_list[0]) + controller[1] + u_list[0]
    #     # u = u_list[0]

    #     pendulum.step(u)
    #     controls[i] = u.squeeze()

    # t = np.ones(N-1) * dt
    # t = np.cumsum(t)
    # plt.plot(t, states[0, :], label='theta')
    # plt.plot(t, states[1, :], label='theta_dot')
    # plt.plot(t, controls, label='control')
    # plt.legend()
    # plt.show()

    # # plotting original ddp solution
    # t = np.ones(N) * dt
    # t = np.cumsum(t)
    # x_arr = np.array(x_list).squeeze()
    # plt.plot(t, x_arr[:-1, 0], label='theta')
    # plt.plot(t, x_arr[:-1, 1], label='theta_dot')
    # plt.plot(t, np.array(u_list).squeeze(), label='control')
    # plt.legend()
    # plt.show()
    # exit()


    # closed loop control
    N = 30
    ddp_horizon = 20
    states = np.zeros((2, N-1))
    controls = np.zeros(N-1)
    for i in range(N-1):
        print i
        state = pendulum.get_state()
        states[:, i] = state.squeeze()

        controller_list, x_list, u_list = ddp.solve(x0, ddp_horizon, dt, quad_step_cost_func, quad_final_cost_func)
        controller = controller_list[0]
        u = np.dot(controller[0], state - x_list[0]) + controller[1] + u_list[0]

        pendulum.step(u)
        controls[i] = u.squeeze()


    t = np.ones(N-1) * dt
    t = np.cumsum(t)



    plt.plot(t, states[0, :], label='theta')
    plt.plot(t, states[1, :], label='theta_dot')
    plt.plot(t, controls, label='control')

    plt.legend()
    plt.show()



