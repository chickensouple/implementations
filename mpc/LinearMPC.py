import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

class HistoryList(object):
    def __init__(self, size, max_hist):
        self.hist = np.zeros((max_hist, size))
        self.max_hist = max_hist
        self.clear()

    def clear(self):
        self.__buffer_len = 0
        self.__erase_idx = 0

    def add(self, state):
        if self.__buffer_len >= self.max_hist:
            self.hist[self.__erase_idx, :] = state
            self.__erase_idx += 1
            if self.__erase_idx >= self.max_entries:
                self.__erase_idx = 0
        else:
            self.hist[self.__buffer_len, :] = state
            self.__buffer_len += 1

    def get_hist(self):
        if self.__buffer_len < self.max_hist:
            return self.hist[:self.__buffer_len, :]
        else:
            list1 = self.hist[self.__erase_idx:, :]
            list2 = self.hist[:self.__erase_idx, :]
            return np.concatenate((list1, list2), axis=0)

class DoubleIntegrator(object):
    """
    Simulates a unit mass moving along a 1D frictionless track
    """
    def __init__(self, max_force=1, max_hist=200):
        self.max_force = max_force

        # q and q_dot
        self.x = np.zeros((2, 1))
        self.hist = HistoryList(4, max_hist)
        self.clear()

    def clear(self):
        self.T = 0
        self.hist.clear()

    def set_state(self, x):
        if x.shape != self.x.shape:
            raise Exception("Shape of states are not the same")
        self.x = x
        self.clear()

    def get_state(self):
        return self.x

    def step(self, control, dt):
        hist_entry = np.array([self.x[0, 0], self.x[1, 0], control, self.T])
        self.hist.add(hist_entry)

        if control > self.max_force:
            control = self.max_force
        elif control < -self.max_force:
            control = self.max_force

        noise = np.random.randn() * 0.1
        self.x[0, 0] = self.x[0, 0] + dt * self.x[1, 0] + (dt**2) * control * 0.5
        self.x[1, 0] = self.x[1, 0] + dt * (control + noise)
        self.T += dt
        return self.x

    def plot(self):
        hist = self.hist.get_hist()

        # plt.subplot(3, 1, 1)
        plt.plot(hist[:, 3], hist[:, 0], label='x', color='g')
        plt.plot(hist[:, 3], hist[:, 1], label='x_dot', color='r')
        plt.plot(hist[:, 3], hist[:, 2], label='u', color='b')
        plt.xlabel('time (seconds)')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # A = np.array([1., ])

    env = DoubleIntegrator()

    # np.random.seed(0)
    x0 = np.random.randn(2, 1)
    env.set_state(x0)
    print "x0: ", x0.T

    T = 20
    dt = 0.1

    
    def solve(x0, T, dt):
        x = cvx.Variable(2, T)
        u = cvx.Variable(1, T)
        A = np.array([[1., dt], [0, 1]])
        B = np.array([[0, 1.]]).T

        Q = np.eye(2)
        R = np.eye(2)

        objective = cvx.Minimize(cvx.sum_squares(x))
        # objective = cvx.Minimize(cvx.norm(x, 1))

        constraints = [x[:, 0] == np.dot(A, x0) + B*u[:, 0],
                       x[:, 1:] == A*x[:, :-1] + B*u[:, 1:],
                       cvx.abs(u) <= 1.]
        prob = cvx.Problem(objective, constraints)
        prob.solve()
        return u.value


    state = x0
    for i in range(100):
        control = solve(state, T, dt)[0, 0]
        state = env.step(control, dt)
        # print control

    env.plot()
    # print "Optimal value", prob.solve()
    # print "Optimal var"
    # print x.value # A numpy matrix.
    # print u.value
