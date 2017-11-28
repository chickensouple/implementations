import numpy as np
import scipy.linalg

# approximates infinite horizon LQR with normal recursive solution
# for discrete time systems
#
# x is dimension m, u is dimension l
# for x_{k+1} = Ax + Bu
# find K for 
# u_k = K x_k
# min x_n'Px_n + sum{x_k' Q x_k + u_k' R u_k} from k = 0 to n-1
# subject to x_{k+1} = A x_{k} + b u_{k}
class LQR(object):
    def __init__(self, A, B, P, Q, R, N):
        self.state_dim = A.shape[1]
        self.control_dim = B.shape[1]

        if A.shape != (self.state_dim, self.state_dim):
            raise Exception('A must be m by m')
        if B.shape != (self.state_dim, self.control_dim):
            raise Exception('B must be m by l')
        if P.shape != (self.state_dim, self.state_dim):
            raise Exception('P must be m by m')
        if Q.shape != (self.state_dim, self.state_dim):
            raise Exception('Q must be m by m')
        if R.shape != (self.control_dim, self.control_dim):
            raise Exception('R must be l by l')


        P_k = np.copy(P)

        for i in range(N):
            K_1 = np.dot(B.T, np.dot(P_k, B)) + R
            K_2 = -np.dot(B.T, np.dot(P_k, A))
            K = np.linalg.solve(K_1, K_2)

            P_k = np.dot(A.T, np.dot(P_k, A)) + Q + \
                np.dot(np.dot(A.T, np.dot(P_k, B)), K)

        self.K = K

    def get_action(self, x0):
        u = np.dot(self.K, x0)
        return u

if __name__ == '__main__':
    A = np.array([[1., 0.1], [0, 1.]])
    # B = np.array([[0, 0.1]]).T
    B = np.array([[0, 0.1, 0.1], [0.1, 0, 0]])
    Q = np.eye(2)
    P = np.eye(2) * 7
    R = np.eye(3)
    x0 = np.array([[2., 2.]]).T

    lqr = LQR(A, B, P, Q, R, 40)
    # u, cost = lqr.solve(x0)

