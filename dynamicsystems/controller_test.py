import numpy as np
import controllers
import models
import math


def angle_diff(angle1, angle2):
    diff = angle1 - angle2
    while diff > np.pi:
        diff -= 2*np.pi
    while diff < -np.pi:
        diff += 2*np.pi
    return diff

def pid_test():
    import matplotlib.pyplot as plt

    dt = 0.05
    pid = controllers.PID(3., 0., -1., 10.)
    pendulum = models.Pendulum(dt=dt)
    pendulum.set_state(np.array([[0.1, 0.1]]).T)

    target = np.array([0.]).T
    N = 100
    states = np.zeros((2, N))
    controls = np.zeros(N)
    for i in range(N):
        state = pendulum.get_state()
        states[:, i] = state.squeeze()

        delta_state = angle_diff(target, state[0])

        u = pid.get_action(delta_state, dt, deriv=state[1])
        u = np.array([u])
        controls[i] = u
        pendulum.step(u)


    t = np.ones(N) * dt
    t = np.cumsum(t)

    plt.plot(t, states[0, :], label='theta')
    plt.plot(t, states[1, :], label='theta_dot')
    plt.plot(t, controls, label='control')
    plt.legend()
    plt.show()


def energy_test():
    import matplotlib.pyplot as plt

    dt = 0.05
    controller = controllers.PendulumEnergySwingup(0.5, 0)
    pendulum = models.Pendulum(dt=dt)
    pendulum.set_state(np.array([[math.pi, 0.1]]).T)

    N = 500
    states = np.zeros((2, N))
    controls = np.zeros(N)
    energies = np.zeros(N)
    for i in range(N):
        state = pendulum.get_state()
        states[:, i] = state.squeeze()
        energy = pendulum.get_energy()
        energies[i] = energy

        u = controller.get_action(state, energy)
        u = np.array([u])

        controls[i] = u

        pendulum.step(u)

    t = np.ones(N) * dt
    t = np.cumsum(t)

    plt.plot(t, states[0, :], label='theta')
    plt.plot(t, states[1, :], label='theta_dot')
    plt.plot(t, controls, label='control')
    plt.plot(t, energies, label='energy')
    plt.legend()
    plt.show()


def hybrid_test():
    import matplotlib.pyplot as plt

    dt = 0.05
    pid = controllers.PID(3., 0., -1., 10.)
    energy_swingup = controllers.PendulumEnergySwingup(0.5, 0)
    controller = controllers.PendulumHybrid(pid, energy_swingup)
    pendulum = models.Pendulum(dt=dt)
    pendulum.set_state(np.array([[math.pi, 0.1]]).T)

    N = 500
    target = np.array([0.]).T
    states = np.zeros((2, N))
    controls = np.zeros(N)
    energies = np.zeros(N)
    for i in range(N):
        state = pendulum.get_state()
        states[:, i] = state.squeeze()

        delta_state = angle_diff(target, state[0])

        energy = pendulum.get_energy()
        energies[i] = energy

        u = controller.get_action(delta_state, dt, state[1], state, energy)
        u = np.array([u])

        controls[i] = u

        pendulum.step(u)

    t = np.ones(N) * dt
    t = np.cumsum(t)

    plt.plot(t, states[0, :], label='theta')
    plt.plot(t, states[1, :], label='theta_dot')
    plt.plot(t, controls, label='control')
    plt.plot(t, energies, label='energy')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # pid_test()
    # energy_test()
    hybrid_test()


