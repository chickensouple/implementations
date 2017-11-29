import numpy as np
from model_base import ModelBase
from lti_system import LTISystem

class DoubleIntegrator(LTISystem):
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
        A = np.array([[0., 1.], [0., 0.]])
        B = np.array([[0., 1.]]).T
        control_limits = [np.array([-max_acc]), np.array([max_acc])]
        super(DoubleIntegrator, self).__init__(A, B, control_limits, **kwargs)

if __name__ == '__main__':
    env = DoubleIntegrator()
    x0 = np.zeros((2, 1))
    u0 = np.zeros((1, 1))
    base, A, B = env.get_linearization(x0, u0)

    print "base:", base
    print "A:", A
    print "B:", B

    A, B = env.discretize(0.1)
    print "A:", A
    print "B:", B

    
    # import matplotlib.pyplot as plt

    # env = DoubleIntegrator(dt = 0.05)
    # controls = [-0.2, -0.2, -0.2, -0.2, -1.2, -1.2, 0, 5., 2., 2., 2.]
    # states = np.zeros((env.state_dim, len(controls)))
    # for idx, control in enumerate(controls):
    #     control = np.array([[control]])
    #     state, _ = env.step(control)
    #     states[:, idx] = state.squeeze()

    # plt.plot(states[0, :])
    # plt.show()


