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
    def __init__(self, L=1., m=1., m_type='point', mu=0., max_torque=1., **kwargs):
        control_limits = [np.array([-max_torque]), np.array([max_torque])]
        super(Pendulum, self).__init__(2, 1, control_limits, **kwargs)
        self.L = L
        self.m = m
        self.m_type = m_type
        self.mu = mu

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
        fric_torque = -x[1] * self.mu

        x_dot = np.zeros(x.shape)
        x_dot[0] = x[1]
        x_dot[1] = (grav_torque + torque + fric_torque) / self.inertia
        return x_dot

    def after_step(self):
        energy = 0.5 * self.inertia * self.x[1] * self.x[1] + \
            self.m * 9.81 * np.cos(self.x[0])
        print "energy: ", energy
        # wrap angle to [-pi, pi)
        while self.x[0] < -np.pi:
            self.x[0] += 2 * np.pi
        while self.x[0] >= np.pi:
            self.x[0] -= 2 * np.pi


if __name__ == '__main__':
    # import matplotlib.pyplot as plt

    # env = Pendulum()
    # env.x = np.array([[0.1, 0]]).T
    # controls = np.array([[0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
    # states = np.zeros((env.state_dim, len(controls)))
    # for idx, control in enumerate(controls):
    #     state, _ = env.step(control)
    #     states[:, idx] = state.squeeze()

    # plt.scatter(states[0, :], states[1, :])
    # plt.show()

    import matplotlib
    matplotlib.use('TkAgg')
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
    from matplotlib.figure import Figure
    import Tkinter as tk
    import sys
    from threading import Lock
    import copy

    class Application(tk.Frame):
        def __init__(self, master=None):
            tk.Frame.__init__(self,master)
            self.dt = 0.02
            self.pendulum = Pendulum(dt=self.dt, mu=0.1)
            # self.pendulum.x = np.array([[0.05, 0]]).T
            self.pendulum.x = np.array([[np.pi, 0]]).T

            self.createWidgets()

            self.control_lock = Lock()
            self.control = np.array([0])
            self.update()


        def on_key_event(self, event):
            if event.key == 'left':
                self.control_lock.acquire()
                self.control[0] = -1.
                self.control_lock.release()
            elif event.key == 'right':
                self.control_lock.acquire()
                self.control[0] = 1.
                self.control_lock.release()

        def draw_pendulum(self):
            theta = self.pendulum.x[0]
            y = np.cos(theta)
            x = -np.sin(theta)

            self.ax.cla()
            self.ax.plot([0, x], [0, y])
            self.ax.axis((-1,1,-1,1))

            self.canvas.draw()

        
        def createWidgets(self):
            fig = plt.figure(figsize=(8,8))
            self.ax = fig.add_subplot(111)

            self.canvas = FigureCanvasTkAgg(fig,master=root)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.canvas.mpl_connect('key_press_event', self.on_key_event)

            self.draw_pendulum()
            self.canvas.show()


        def update(self):
            self.control_lock.acquire()
            control = copy.deepcopy(self.control)
            self.control[0] = 0
            self.control_lock.release()
            self.pendulum.step(control)
            self.draw_pendulum()
            # refresh every 0.1 seconds
            self.after(int(self.dt * 1e3), self.update)


    root=tk.Tk()
    app=Application(master=root)
    app.mainloop()



