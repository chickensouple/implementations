import numpy as np
import math
from model_base import ModelBase

class Pendulum(ModelBase):
    """
    Models a simple pendulum driven by a motor
    
    state is x = [theta, theta_dot]
    where theta is 0 when pointing upwards
    control input is u = [torque]
    
    Args:
        x (numpy array): array of length 2 of current state
        u (numpy array): array of length 1 of current inputs
        L (float, optional): length of pendulum
        m (float, optional): mass of pendulum
        m_type (str, optional): type of pendulum. 'point' for a point mass at end of pendulum
        or 'rod' for a mass evenly distributed across a solid rod
        max_torque (float, optional): maximum torque that can be applied by motor
    
    Returns:
        numpy array: array of length 2 of derivative 
    """
    def __init__(self, L=1., m=1., m_type='point', max_torque=1., **kwargs):
        control_limits = [np.array([-max_torque]), np.array([max_torque])]
        super(Pendulum, self).__init__(2, 1, control_limits, **kwargs)
        self.L = L
        self.m = m
        self.m_type = m_type

        # compute center of mass and moment of inertia
        if m_type == 'point':
            self.com = L
            self.inertia = m * L * L
        elif m_type == 'rod':
            self.com = 0.5 * L
            self.inertia = m * L * L / 3
        else:
            raise Exception('Not a valid m_type')


    def diff_eq(self, x, u):
        u = self._check_and_clip(x, u)
        torque = u[0]

        grav = 9.81
        grav_torque = self.m * grav * self.com * np.sin(x[0])

        x_dot = np.zeros(x.shape)
        x_dot[0] = x[1]
        x_dot[1] = (grav_torque + torque) / self.inertia
        return x_dot

    def after_step(self):
        print("hi")


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = Pendulum()
    env.x = np.array([[0.1, 0]]).T
    controls = np.array([[0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
    states = np.zeros((env.state_dim, len(controls)))
    for idx, control in enumerate(controls):
        state, _ = env.step(control)
        states[:, idx] = state.squeeze()

    plt.scatter(states[0, :], states[1, :])
    plt.show()

    # import matplotlib
    # matplotlib.use('TkAgg')

    # from numpy import arange, sin, pi
    # from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
    # # implement the default mpl key bindings
    # from matplotlib.backend_bases import key_press_handler
    # from matplotlib.figure import Figure

    # import sys
    # if sys.version_info[0] < 3:
    #     import Tkinter as Tk
    # else:
    #     import tkinter as Tk

    # root = Tk.Tk()
    # root.wm_title("Pendulum")


    # f = Figure(figsize=(5, 4), dpi=100)
    # a = f.add_subplot(111)


    # t = arange(0.0, 3.0, 0.01)
    # s = sin(2*pi*t)

    # a.plot(t, s)


    # # a tk.DrawingArea
    # canvas = FigureCanvasTkAgg(f, master=root)
    # canvas.show()
    # canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    # toolbar = NavigationToolbar2TkAgg(canvas, root)
    # toolbar.update()
    # canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)


    # def on_key_event(event):
    #     print('you pressed %s' % event.key)
    #     key_press_handler(event, canvas, toolbar)

    # canvas.mpl_connect('key_press_event', on_key_event)


    # def _quit():
    #     root.quit()     # stops mainloop
    #     root.destroy()  # this is necessary on Windows to prevent
    #                     # Fatal Python Error: PyEval_RestoreThread: NULL tstate

    # button = Tk.Button(master=root, text='Quit', command=_quit)
    # button.pack(side=Tk.BOTTOM)

    # Tk.mainloop()

