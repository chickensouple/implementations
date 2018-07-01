import numpy as np
import dynamicsystems.controllers
import dynamicsystems.models
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

    dt = 0.01
    pid = dynamicsystems.controllers.PID(5., 0.02, -1., 10.)
    pendulum = dynamicsystems.models.Pendulum(dt=dt)
    pendulum.set_state(np.array([[0.1, 0.1]]).T)

    target = np.array([0.2]).T
    N = 500
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


def pid_rocket_test():
    import matplotlib.pyplot as plt

    dt = 0.05
    pid = dynamicsystems.controllers.PID(-3., -0.01, 5., 100.)
    pendulum = dynamicsystems.models.Rocket(dt=dt)

    target = np.array([100]).T
    N = 1600
    states = np.zeros((3, N))
    controls = np.zeros(N)
    for i in range(N):
        state = pendulum.get_state()
        states[:, i] = state.squeeze()

        delta_state = target - state[0]

        u = pid.get_action(delta_state, dt, deriv=state[1])
        u = np.array([u])
        controls[i] = u
        pendulum.step(u)


    t = np.ones(N) * dt
    t = np.cumsum(t)

    plt.plot(t, states[0, :], label='height')
    plt.plot(t, states[1, :], label='velocity')
    plt.plot(t, states[2, :] * 100 / states[2, 0], label='fuel')
    plt.plot(t, controls, label='control')

    plt.legend()
    plt.show()





def energy_test():
    import matplotlib.pyplot as plt

    dt = 0.05
    controller = dynamicsystems.controllers.PendulumEnergySwingup(0., 2., 0.001, 0.)
    pendulum = dynamicsystems.models.Pendulum(dt=dt)
    pendulum.set_state(np.array([[math.pi, 0.]]).T)

    N = 500
    states = np.zeros((2, N))
    controls = np.zeros(N)
    energies = np.zeros(N)
    for i in range(N):
        state = pendulum.get_state()
        states[:, i] = state.squeeze()
        energy = pendulum.get_energy()
        energies[i] = energy

        u = controller.get_action(state, energy, dt)
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

    dt = 0.1
    pid = dynamicsystems.controllers.PID(2., 0., -1.)
    energy_swingup = dynamicsystems.controllers.PendulumEnergySwingup(0., 2., 0.0001, 0.)
    controller = dynamicsystems.controllers.PendulumHybrid(pid, energy_swingup)
    pendulum = dynamicsystems.models.Pendulum(dt=dt)
    pendulum.set_state(np.array([[math.pi, 0.]]).T)

    N = 200
    target = np.array([0.]).T
    states = np.zeros((2, N))
    controls = np.zeros(N)
    energies = np.zeros(N)
    control_types = np.zeros(N)
    for i in range(N):
        state = pendulum.get_state()
        states[:, i] = state.squeeze()

        delta_state = angle_diff(target, state[0])

        energy = pendulum.get_energy()
        energies[i] = energy

        u, control_type = controller.get_action(delta_state, dt, state[1], state, energy)
        u = np.array([u])

        controls[i] = u
        control_types[i] = control_type

        pendulum.step(u)

    t = np.ones(N) * dt
    t = np.cumsum(t)

    plt.plot(t, states[0, :], label='theta')
    plt.plot(t, states[1, :], label='theta_dot')
    plt.plot(t, controls, label='control')
    plt.plot(t, energies, label='energy')
    plt.plot(t, control_types, label='controller')
    plt.legend()
    plt.show()


def lqr_batch_test():
    import matplotlib.pyplot as plt
    dt = 0.05
    N = 200
    sys = dynamicsystems.models.DoubleIntegrator(dt=dt)

    x0 = np.array([[2., 2.]]).T
    A, B = sys.discretize(dt)
    Q = np.eye(2) * 4
    P = Q
    R = np.eye(1) * 1

    sys.set_state(x0)

    controller = dynamicsystems.controllers.LQRBatch(A, B, Q, P, R, 30)

    states = np.zeros((2, N-1))
    controls = np.zeros(N-1)
    for i in range(N-1):
        state = sys.get_state()
        states[:, i] = state.squeeze()

        u, cost = controller.solve(state)

        control = u[0, 0]
        controls[i] = control

        sys.step(np.array([[control]]))

    t = np.ones(N-1) * dt
    t = np.cumsum(t)

    plt.plot(t, states[0, :], label='x')
    plt.plot(t, states[1, :], label='x_dot')
    plt.plot(t, controls, label='control')

    plt.legend()
    plt.show()


def lqr_test():
    import matplotlib.pyplot as plt
    dt = 0.05
    N = 200
    sys = dynamicsystems.models.DoubleIntegrator(dt=dt)

    x0 = np.array([[2., 2.]]).T
    A, B = sys.discretize(dt)
    Q = np.eye(2) * 4
    P = Q
    R = np.eye(1) * 1

    sys.set_state(x0)

    controller = dynamicsystems.controllers.LQR(A, B, Q, P, R, 100)

    states = np.zeros((2, N-1))
    controls = np.zeros(N-1)
    for i in range(N-1):
        state = sys.get_state()
        states[:, i] = state.squeeze()

        u = controller.get_action(state)

        control = u[0, 0]
        controls[i] = control

        sys.step(np.array([[control]]))

    t = np.ones(N-1) * dt
    t = np.cumsum(t)

    plt.plot(t, states[0, :], label='x')
    plt.plot(t, states[1, :], label='x_dot')
    plt.plot(t, controls, label='control')

    plt.legend()
    plt.show()

def lqr_test_nonzero():
    import matplotlib.pyplot as plt
    dt = 0.05
    N = 400
    sys = dynamicsystems.models.DoubleIntegrator(dt=dt)

    x0 = np.array([[2., 2.]]).T
    A, B = sys.discretize(dt)
    Q = np.eye(2) * 4
    P = Q
    R = np.eye(1) * 1

    sys.set_state(x0)
    controller = dynamicsystems.controllers.LQR(A, B, Q, P, R, 100)

    target_state = np.array([[-2, 0]]).T
    target_control = np.array([[0]])


    states = np.zeros((2, N-1))
    controls = np.zeros(N-1)
    for i in range(N-1):
        state = sys.get_state()
        states[:, i] = state.squeeze()

        delta_state = state - target_state

        u = controller.get_action(delta_state)

        control = u[0, 0]
        controls[i] = control

        sys.step(np.array([[control]]) - target_control)

    t = np.ones(N-1) * dt
    t = np.cumsum(t)

    plt.plot(t, states[0, :], label='x')
    plt.plot(t, states[1, :], label='x_dot')
    plt.plot(t, controls, label='control')

    plt.legend()
    plt.show()


if __name__ == '__main__':

    # pid_test()
    pid_rocket_test()
    # energy_test()
    # hybrid_test()
    # lqr_batch_test()
    # lqr_test()
    # lqr_test_nonzero()
