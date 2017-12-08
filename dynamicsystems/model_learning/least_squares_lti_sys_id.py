import numpy as np

def least_squares_lti_sys_id(x, u, x_next):
    """
    Performs Least Squares system identification on
    Discrete Linear Time invariant systems of the form
    
    x_next = A*x + B*u
    
    with state dimensions n
    control dimensions m
    and number of observations l

    For exact solution, l should be equal to n+m
    Any less, and the solution will be under constrained
    Any more, and the solution will be over constrained

    Args:
        x (np array): size (n, l) array of states
        u (np array): size (m, l) array of actions
        x_next (np array): size (n, l) array of next statse
    """
    n = x.shape[0]
    m = u.shape[0]
    l = x.shape[1]
    if x_next.shape != (n, l):
        raise Exception('x_next must be (n, l)')
    if x.shape != (n, l):
        raise Exception('x must be (n, l)')
    if u.shape != (m, l):
        raise Exception('u must be (m, l)')

    obs_vec = x_next.T.flatten()
    obs_vec = np.reshape(obs_vec, (-1, 1))

    mat = np.zeros((n*l, n*n + n*m))

    count = 0
    for i in range(l):
        for j in range(n):
            idx1 = j*n
            idx2 = (j+1)*n
            mat[count, idx1:idx2] = x[:, i]
            idx1 = n*n + j*m
            idx2 = n*n + (j+1)*m
            mat[count, idx1:idx2] = u[:, i]
            count += 1

    params, _, _, _ = np.linalg.lstsq(mat, obs_vec)

    A_est = np.reshape(params[:n*n], (n, n))
    B_est = np.reshape(params[n*n:], (n, m))

    return A_est, B_est


if __name__ == '__main__':
    n = 3
    m = 2

    A = np.random.random((n, n))
    B = np.random.random((n, m))

    l = 6
    x = np.random.random((n, l))
    u = np.random.random((m, l))
    x_next = np.dot(A, x) + np.dot(B, u)

    A_est, B_est = least_squares_lti_sys_id(x, u, x_next)

    print('True A:\n' + str(A))
    print('True B:\n' + str(B))
    print('Est. A:\n' + str(A_est))
    print('Est. B:\n' + str(B_est))
    print('Max |A - Est. A|: ' + str(np.max(np.abs(A - A_est))))
    print('Max |B - Est. B|: ' + str(np.max(np.abs(B - B_est))))

