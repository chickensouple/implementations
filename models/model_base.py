import numpy as np
from ode_numerical import ode_solver_once

class ModelBase(object):
    def __init__(self, state_dim, control_dim, control_limits, dt=0.05):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.control_limits = control_limits
        self.dt = dt

        if len(self.control_limits) != 2:
            raise Exception('Control Limits should consist of lower and upper bounds')

        if len(self.control_limits[0]) != self.control_dim or \
            len(self.control_limits[1]) != self.control_dim:
            raise Exception('Control Limits not the right size')

        self.reset()

    def reset(self):
        self.x = np.zeros((self.state_dim, 1))
        self.T = 0

    def _check_and_clip(self, x, u):
        if len(x) != self.state_dim:
            raise Exception("State is not proper shape")
        if len(u) != self.control_dim:
            raise Exception("Control input is not proper shape")

        u = np.clip(u, self.control_limits[0], self.control_limits[1])
        return u


    def get_diff_eq(self):
        """
        returns a function to calculate diff_eq
        without needing to reference the object
        
        Returns:
            function: differential equation
        """
        func = lambda x, u: self.diff_eq(x, u)
        return func

    def step(self, u):
        f = self.get_diff_eq()
        self.x = ode_solver_once(f, self.x, u, self.dt)
        self.T += self.dt
        self.after_step()
        return self.x, self.T

    def after_step(self):
        """
        Overload this function if you need to do some post processing
        after the ode solver is called
        """
        pass



