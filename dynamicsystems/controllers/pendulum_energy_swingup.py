import numpy as np
import math

class PendulumEnergySwingup(object):
    def __init__(self, k, E0):
        self.k = k
        self.E0 = E0

    def get_action(self, state, current_energy):
        diff_energy = current_energy - self.E0
        control = self.k * diff_energy * state[1] * math.cos(state[0])
        return control


