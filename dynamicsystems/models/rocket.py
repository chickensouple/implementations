import numpy as np
from model_base import ModelBase

class Rocket(ModelBase):
    """
    Models a simple rocket

    state is [h, v, m]
    where h is height, v is velocity, m is mass of fuel
    control input is [rate of mass explusion] in mass / s
    """
    def __init__(self, ve=150., max_u=10., mass_r=1., **kwargs):
        """
        Initializes a simple rocket
        
        Args:
            ve (float, optional): magnitude of expulsion velocity (positive number)
            max_u (float, optional): magnitude of maximum mass expulsion rate (positive number)
            mass_r (float, optional): mass of rocket not counting fuel
            **kwargs: Description
        """
        control_limits = [np.array([-max_u]), np.array([0])]
        super(Rocket, self).__init__(3, 1, control_limits, **kwargs)
        self.ve = -ve
        self.mass_r = mass_r

    def reset(self):
        super(Rocket, self).reset()
        self.set_state(np.array([[0, 0, 100.]]).T)

    def after_step(self):
        if self.x[0] < 0:
            self.x[0] = 0
            self.x[1] = 0

    def diff_eq(self, x, u):
        u = self._check_and_clip(x, u)

        x_dot = np.zeros(x.shape)
        x_dot[0] = x[1]

        # if there is no more fuel, don't 
        # add acceleration from fuel explusion
        if x[2] <= 0:
            x_dot[1] = -9.81
            x_dot[2] = 0
        else:
            x_dot[1] = -9.81 + u * self.ve / (x[2] + self.mass_r)
            x_dot[2] = u

        # if rocket is on ground, don't let it accelerate downwards
        if x[0] <= 0:
            if x_dot[1] <= 0:
                x_dot[1] = 0

        return x_dot

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dt = 0.1
    env = Rocket(dt=dt)
    controls = [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5]
    states = np.zeros((env.state_dim, len(controls)))
    for idx, control in enumerate(controls):
        control = np.array([[control]])
        state, _ = env.step(control)
        states[:, idx] = state.squeeze()

    t = np.ones(len(controls)) * dt
    t = np.cumsum(t)

    plt.plot(t, states[0, :], label='height')
    plt.plot(t, states[1, :], label='velocity')
    plt.plot(t, states[2, :] / states[2, 0], label='mass')
    plt.legend()
    plt.show()
