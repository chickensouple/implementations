import numpy as np
from model_base import ModelBase


class DoubleIntegrator(ModelBase):
    """
    Models a simple double integrator
    
    state for system is [x, x_dot]
    control input is [acceleration]
    """
    def __init__(self, max_acc=1., **kwargs):
        """
        Initializes a Double Integrator
        
        Args:
            max_acc (float, optional): maximum acceleration in m/s/s
            **kwargs: Description
        """
        control_limits = [np.array([-max_acc]), np.array([max_acc])]
        super(DoubleIntegrator, self).__init__(2, 1, control_limits, **kwargs)

    def diff_eq(self, x, u):
        u = self._check_and_clip(x, u)
        accel = u[0]

        x_dot = np.zeros(x.shape)
        x_dot[0] = x[1]
        x_dot[1] = accel
        return x_dot


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = DoubleIntegrator()
    controls = np.array([[-0.2, -0.2, -0.2, -1.2, 0, 5., 2.]]).T
    states = np.zeros((env.state_dim, len(controls)))
    for idx, control in enumerate(controls):
        state, _ = env.step(control)
        states[:, idx] = state.squeeze()

    plt.plot(states[0, :])
    plt.show()


