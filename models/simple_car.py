import numpy as np
from model_base import ModelBase
import math

class SimpleCar(ModelBase):
    """
    Models a simple car with front wheel to back wheel of length L
    Car utilizes front wheel steering with max turn angle
    
    state is x x = [x, y, v, theta]
    where (x, y) is a 2d coordinate, v is the forward velocity of the car
    and theta is the heading of the car (where 0 is pointed in positive x direction)
    
    control input is u = [a, phi]
    where a is acceleration of the car and phi is turning angle
    
    Args:
        x (numpy array): array of length 4 of current state
        u (numpy array): array of length 2 of current inputs
        L (float, optional): front wheel to back wheel distance (default: 1)
        max_accel (float, optional): maximum acceleration of car
        max_turn (float, optional): maximum turn angle of car
    
    Returns:
        numpy array: array of length 4 of derivative 
    """
    def __init__(self, L=1., max_accel=1., max_turn=np.pi/3, **kwargs):
        control_limits = [np.array([-max_accel, -max_turn]), np.array([max_accel, max_turn])]
        super(SimpleCar, self).__init__(4, 2, control_limits, **kwargs)
        self.L = L

    def diff_eq(self, x, u):
        u = self._check_and_clip(x, u)

        accel = u[0]
        ang_accel = u[1]

        x_dot = np.zeros(x.shape)
        x_dot[0] = x[2] * math.cos(x[3])
        x_dot[1] = x[2] * math.sin(x[3])
        x_dot[2] = accel
        x_dot[3] = -x[2] / self.L * math.tan(u[1])
        return x_dot


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = SimpleCar(dt=0.1)
    controls = \
        np.array([[1., 0],
                  [1., 0],
                  [1., 0.5],
                  [1., 0.5],
                  [0, 0.5],
                  [0, 0.5],
                  [0, 0.5],
                  [0, 0.5],
                  [-1, 0],
                  [-1, 0],
                  [-1, 0],
                  [-1, 0],
                  [0, 0]])
    states = np.zeros((env.state_dim, len(controls)))
    for idx, control in enumerate(controls):
        state, _ = env.step(control)
        states[:, idx] = state.squeeze()

    plt.scatter(states[0, :], states[1, :])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
