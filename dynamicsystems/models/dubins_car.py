import numpy as np
from . model_base import ModelBase

class DubinsCar(ModelBase):
    """
    Models a dubins car where the only input is angular velocity of car

    state is [x, y, theta]
    where (x, y) is a 2D coordinate of the car
    and theta is the heading of the car (where 0 is pointed in positive x direction)
    control input is [angular_velocity]
    """
    def __init__(self, v=1., max_w=1., **kwargs):
        """
        Initializes DubinsCar
        
        Args:
            v (float, optional): forward velocity of car in m/s
            max_w (float, optional): maximum angular velocity of car in rad/s
            **kwargs: Description
        """
        control_limits = [np.array([-max_w]), np.array([max_w])]
        super(DubinsCar, self).__init__(3, 1, control_limits, **kwargs)
        self.v = v

    def diff_eq(self, x, u):
        u = self._check_and_clip(x, u)
        ang_vel = u[0]

        x_dot = np.zeros(x.shape)
        x_dot[0] = self.v * np.cos(x[2])
        x_dot[1] = self.v * np.sin(x[2])
        x_dot[2] = ang_vel
        return x_dot

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = DubinsCar()
    controls = np.array([[-0.2, -0.2, -0.2, 0, 0, 0.4, 0.4]]).T
    states = np.zeros((env.state_dim, len(controls)))
    for idx, control in enumerate(controls):
        state, _ = env.step(control)
        states[:, idx] = state.squeeze()

    plt.scatter(states[0, :], states[1, :])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
