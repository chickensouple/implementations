import numpy as np


class PendulumHybrid(object):
    def __init__(self, pid, energy_swingup):
        self.pid = pid
        self.energy_swingup = energy_swingup

    def get_action(self, delta_state, dt, deriv, state, current_energy):
        if abs(current_energy) < 1 \
            and abs(state[0]) < 1 \
            and abs(state[1]) < 1:
            control = self.pid.get_action(delta_state, dt, deriv=deriv)
        else:
            control = self.energy_swingup.get_action(state, current_energy)

        return control


