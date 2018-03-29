import numpy as np
from matplotlib import pyplot as plt



class Linear_model(object):


    # Constructor takes weights and noise variance
    def __init__(self, w, sigma_2):
        self.w = w
        self.sigma = sigma_2

    def ground_truth(self, x):
        return self.w.T*x

    # Generate vectorized labels
    def generate_labels(self, x):

        y = self.w.T*x + np.random.multivariate_normal(np.zeros(x.shape[0]), self.sigma * np.identity(x.shape[0]))
        return y


class GP(object):

    # Assume standard RBF Kernel
    def __init__(self, sigma_f, sigma_w, length_scale = 1):
        self.X = []
        self.y  = []
        self.K = []
        self.sigma_f = sigma_f
        self.sigma_w = sigma_w
        self.length_scale = length_scale


    def add_data(self, X, y):
        self.X = X
        self.y = y


    def predict_labels(self, x):
        # First compute Kernel matrix K
        X_X = np.tile(self.X, (self.X.shape[0], 1))
        X_X = (X_X - X_X.T) ** 2
        from scipy.spatial.distance import pdist, cdist, squareform
        X_X = squareform(pdist(self.X.reshape(-1,1), 'euclidean'))
        K = sigma_f**2 *np.exp(-1/2 * X_X / length_scale)

        # Now compute test_data matrix K*
        X_s = cdist(x.reshape(-1,1), self.X.reshape(-1,1))
        K_s = sigma_f**2 * np.exp(-1/2 * X_s / length_scale)

        # Compute the pure test matrix K**
        X_ss = squareform(pdist(x.reshape(-1,1), 'euclidean'))
        K_ss = sigma_f**2 *np.exp(-1/2 * X_ss / length_scale)

        # Compute Mean
        inverse_mat = np.linalg.inv(K + sigma_w**2 * np.identity(self.X.shape[0]))
        mean =  np.dot(np.dot(K_s, inverse_mat), self.y.reshape(-1,1))

        # Compute Covariance
        cov = K_ss - np.dot(K_s, np.dot(inverse_mat, K_s.T))

        return mean, cov

if __name__ == '__main__':

    w = np.array([.5])
    sigma_2 = np.array([1e-2])
    linear_model = Linear_model(w, sigma_2)


    # Acquire Data
    X = np.array([1, 2, 3, 4])
    y = linear_model.generate_labels(X)
    print "Labels y= ", y


    # Build GP Model
    length_scale = 1
    sigma_f = 1 # Signal Noise
    sigma_w = 1e-5 # Model Noise
    GP = GP(sigma_f, sigma_w, length_scale)
    GP.add_data(X, y)

    # Test data
    x = np.linspace(0,10,1000)
    [y_hat, cov] = GP.predict_labels(x)
    sigma = cov #np.atleast_2d(cov.diagonal())
    print "Predicted y=", y_hat

    # Ground truth function
    y_t = linear_model.ground_truth(x)


    # Generate Plots
    fig = plt.figure()

    plt.plot(x, y_t, 'r:', label=u'$f(x) = x$') # Ground Truth
    plt.plot(X, y, 'r.', markersize=10, label=u'Observations') # Plot the observed datapoints
    plt.plot(x, y_hat, 'b-', label=u'Prediction') # Plot the predictions
    plt.fill(np.concatenate([x, x[::-1]]), # Plot Covariance. This basically plots a bounding box around x values
             np.concatenate([y_hat - 1.9600 * sigma,
                             (y_hat+ 1.9600 * sigma)[::-1]]),
             alpha=.1, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.title('Linear Regression with Gaussian Processes')

   # plt.legend(loc='upper left')
    # TODO Fix labeling

    plt.show()