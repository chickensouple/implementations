import numpy as np
import copy

def ode_solver(f, x0, t0, dt, stype='runge_kutta4'):
    """
    numerically computes ode solution
    
    Args:
        f (function): ode to solve, x_dot = f(t, x)
        x0 (numpy): a state_size by 1 numpy array for initial state
        t0 (array): initial time
        dt (TYPE): a length N array of timesteps to integrate over
        stype (str, optional): type of solver, 'euler' or 'runge_kutta4'
    
    Returns:
        x, t: 
        where x is a state_size by N array of states
        t is a length N array of time
    """
    if stype == 'euler':
        func = euler
    elif stype == 'runge_kutta4':
        func = runge_kutta4

    n = len(dt)

    x = np.zeros((len(x0), n+1))
    t = np.zeros(n+1)
    x[:, 0] = x0.squeeze()
    t[0] = t0
    for i in range(1, n+1):
        x_prev = np.array([x[:, i-1]]).T
        x[:, i] = func(f, x_prev, t[i-1], dt[i-1]).squeeze()
        t[i] = t[i-1] + dt[i-1]
    return x, t

def ode_solver_once(f, x, u, dt, stype='runge_kutta4'):
    """
    Numerically computes solution to a time invariant ode with control input
    
    Args:
        f (function): ode to solve, x_dot = f(x, u)
        x (numpy array): a state_size by 1 numpy array for state
        u (numpy array): a control_size by 1 numpy array for control input
        dt (float): time step to integrate over
        stype (str, optional): type of solver, 'euler' or 'runge_kutta4'
    
    Returns:
        numpy array: state_size by 1 numpy array for new state
    """
    if stype == 'euler':
        func = euler
    elif stype == 'runge_kutta4':
        func = runge_kutta4

    f_2 = lambda t, x: f(x, u)
    return func(f_2, x, 0, dt)

def euler(f, x, t, dt):
    # integrates an ode x_dot = f(t, x)
    # over one time step dt
    # returns x(t+dt)
    x_new = x + f(t, x) * dt
    return x_new

def runge_kutta4(f, x, t, dt):
    # integrates an ode x_dot = f(t, x)
    # over one time step dt
    # returns x(t+dt)
    k1 = f(t, x)
    k2 = f(t, x+k1*dt*0.5)
    k3 = f(t, x+k2*dt*0.5)
    k4 = f(t, x+k3*dt)

    slope = (1.0 / 6) * (k1 + 2*(k2 + k3) + k4)
    x_new = x + slope * dt
    return x_new


def control_wrapper(f, u):
    """
    Generates a wrapper function
    that takes in (t, x) as arguments
    to be used in an ode solver
    Args:
        f (function): function that describes dynamics of system
        x_dot = f(x, u)
        first argument must be state, second is control input
        u (function): function that takes in state, time, and outputs 
        a control output, u(x, t)
    """
    f_new = lambda t, x: f(x, u(x, t))
    return f_new

def step_controller(u, dt):
    """
    Generates function u(t) that returns the ith elemnt of u
    at the ith timestep for dt amount of time, creating a step function for u
    
    Args:
        u (array): array of control inputs
        dt (float): timestep size
    """
    def func(x, t, u=copy.deepcopy(u), dt=dt):
        n = len(u)
        if t < 0 or t >= n * dt:
            return 0.
        idx = int(np.floor(t / dt))
        return u[idx]
    return func

def constant_controller(u):
    """
    Generates a function u(t) that returns a constant value
    """
    return lambda x, t: u


# if __name__ == '__main__':
#     from dubins_car import DubinsCar
#     import matplotlib.pyplot as plt
#     car = DubinsCar(1., 1.)

#     dt = 0.1
#     controls = np.array([-0.2, -0.2, -0., 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
#     control_func = step_controller(controls, dt)
#     ode_func = control_wrapper(DubinsCar.diff_eq, control_func)

#     dts = np.ones(len(controls)) * dt
#     x1, t1 = ode_solver(ode_func, car.x, 0, dts)
#     x2, t2 = ode_solver(ode_func, car.x, 0, dts, stype='euler')

#     plt.scatter(x1[0, :], x1[1, :], label='runge_kutta')
#     plt.scatter(x2[0, :], x2[1, :], label='euler')

#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()

#     plt.show()
#     