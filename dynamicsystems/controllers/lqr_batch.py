import numpy as np
import scipy.linalg

# solves LQR in batch solution
# for discrete time systems
#
# x is dimension m, u is dimension l
# for x_{k+1} = Ax + Bu
# find u_0, ..., u_{n-1}
# to solve
# min x_n'Px_n + sum{x_k' Q x_k + u_k' R u_k} from k = 0 to n-1
# subject to x_{k+1} = A x_{k} + b u_{k}
class LQRBatch(object):
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

        S_x = np.zeros((N*self.state_dim, self.state_dim))

        A_pow = np.eye(self.state_dim)
        for i in range(N):
            S_x[i*self.state_dim:(i+1)*self.state_dim, :] = A_pow
            A_pow = np.dot(A, A_pow)

        S_u = np.zeros((N*self.state_dim, (N-1)*self.control_dim))
        B_mat = np.copy(B)
        for i in range(1, N):
            row_1 = i*self.state_dim
            row_2 = (i+1)*self.state_dim
            col_1 = 0
            col_2 = self.control_dim
            for j in range(N-i):
                S_u[row_1:row_2, col_1:col_2] = B_mat
                row_1 += self.state_dim
                row_2 += self.state_dim
                col_1 += self.control_dim
                col_2 += self.control_dim

            B_mat = np.dot(A, B_mat)

        Q_list = [Q for _ in range(N-1)]
        Q_list.append(P)
        Q_mat = scipy.linalg.block_diag(*Q_list)

        R_mat = scipy.linalg.block_diag(*[R for _ in range(N-1)])


        self.H = np.dot(S_u.T, np.dot(Q_mat, S_u)) + R_mat
        self.F = np.dot(S_x.T, np.dot(Q_mat, S_u))
        self.Y = np.dot(S_x.T, np.dot(Q_mat, S_x))

        # controller of form u = K*x0
        self.K = -np.dot(np.linalg.inv(self.H), self.F.T)

        # to solve for one x0
        # u = np.linalg.solve(H, -np.dot(F.T, x0))

    def solve(self, x0):
        if x0.shape != (self.state_dim, 1):
            raise Exception('x0 must be m by 1')
        u = np.dot(self.K, x0)
        cost = np.dot(x0.T, np.dot(self.F, u)) + np.dot(x0.T, np.dot(self.Y, x0))
        cost = np.asscalar(cost)

        # reshape (m * N-1 by 1) into (m by N-1)
        u = np.reshape(u, (-1, self.control_dim)).T
        return u, cost
        
if __name__ == '__main__':
    A = np.array([[1., 0.1], [0, 1.]])
    # B = np.array([[0, 0.1]]).T
    B = np.array([[0, 0.1, 0.1], [0.1, 0, 0]])
    Q = np.eye(2)
    P = np.eye(2) * 7
    R = np.eye(3)
    x0 = np.array([[2., 2.]]).T

    lqr = LQRBatch(A, B, P, Q, R, 5)
    u, cost = lqr.solve(x0)

