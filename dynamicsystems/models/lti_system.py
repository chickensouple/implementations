import numpy as np
from model_base import ModelBase

class LTISystem(ModelBase):
    """
    Generic Linear Time Invariant System of the form
    x_dot = Ax + Bu
    """
    def __init__(self, A, B, control_limits=None, **kwargs):
        if A.shape[0] != B.shape[0]:
            raise Exception('Number of states do not match')

        state_dim = A.shape[0]
        control_dim = B.shape[1]
        if control_limits == None:
            min_limit = np.ones(control_dim) * -np.Inf
            max_limit = np.ones(control_dim) * np.Inf
            control_limits = [min_limit, max_limit]
        super(LTISystem, self).__init__(state_dim, control_dim, control_limits, **kwargs)

        self.A = A
        self.B = B

    def diff_eq(self, x, u):
        u = self._check_and_clip(x, u)

        x_dot = np.dot(self.A, x) + np.dot(self.B, u)
        return x_dot

    def get_linearization(self, x0, u0):
        f0 = self.diff_eq(x0, u0)
        return f0, self.A, self.B

    def discretize(self, dt):
        A_disc = (self.A * dt + np.eye(self.state_dim))
        B_disc = (self.B * dt)
        return A_disc, B_disc


if __name__ == '__main__':
    A = np.array([[1, 2., 0], [0, 1., 0.5], [0, 1., 0]])
    B = np.array([[0, 1, 2.]]).T
    env = LTISystem(A, B)
    x0 = np.zeros((3, 1))
    u0 = np.zeros((1, 1))
    base, A, B = env.get_linearization(x0, u0)

    print "base:", base
    print "A:", A
    print "B:", B

