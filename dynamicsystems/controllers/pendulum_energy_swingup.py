import numpy as np
import math
from pid import PID

class PendulumEnergySwingup(object):
    def __init__(self, E0, kp, ki=0, kd=0):
        self.E0 = E0
        self.pid = PID(kp, ki, kd)

    @staticmethod
    def __get_sign(num):
        if num >= 0:
            return 1
        else:
            return -1

    def get_action(self, state, current_energy, dt):
        diff_energy = self.E0 - current_energy

        action = self.pid.get_action(diff_energy, dt)
        control = PendulumEnergySwingup.__get_sign(state[1]) * action

        return control
