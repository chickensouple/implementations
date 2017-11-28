import numpy as np


class PID(object):
    def __init__(self, kp, ki, kd, int_limit=np.Inf):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.int_limit = int_limit
        self.reset()

    def reset(self):
        self.int = 0
        self.prev_delta_state = 0

    def get_action(self, delta_state, dt, deriv=None):
        control = delta_state * self.kp

        if deriv == None:
            deriv = (delta_state - self.prev_delta_state) / dt
        control += deriv * self.kd
        self.prev_delta_state = delta_state

        self.int += delta_state
        if self.int > self.int_limit:
            self.int = self.int_limit
        if self.int < -self.int_limit:
            self.int = -self.int_limit

        control += self.int * self.ki

        return control


# TODO:
# LQR
# DDP
# ILQG

