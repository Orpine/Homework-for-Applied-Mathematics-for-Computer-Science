import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from cvxopt import solvers, matrix


def qpsub(H, c, A, b):
    if A.size == 0:
        x = np.linalg.pinv(H) * -c
        # x = np.linalg.solve(H, -c)
        l = None
    else:
        M = np.hstack((H, A.T))
        M = np.vstack((M, np.hstack((A, np.matlib.zeros((A.shape[0], M.shape[1] - A.shape[1]))))))
        X = np.linalg.pinv(M) * np.vstack((-c, -b))
        # X = np.linalg.solve(M, np.vstack((-c, -b)))
        x = X[:H.shape[0]]
        l = -X[H.shape[0]:]
    return x, l


def quadprog(H, c, A, b):
    x = np.matrix(solvers.lp(matrix(np.zeros(c.shape)), matrix(-A), matrix(-b))['x'])
    # x = scipy.optimize.linprog(np.ones(c.shape).T[0], -np.array(A), -np.array(b).T[0])['x']
    # x = np.matrix(x).T
    workset = np.array(A * x <= b).T[0]
    max_iter = 200
    for k in xrange(max_iter):
        Aeq = A[np.where(workset == True)]
        g_k = H * x + c
        p_k, lambda_k = qpsub(H, g_k, Aeq, np.matlib.zeros((Aeq.shape[0], 1)))
        if np.linalg.norm(p_k) <= 1e-9:
            if np.min(lambda_k) > 0:
                break
            else:
                pos = np.argmin(lambda_k)
                for i in xrange(b.size):
                    if workset[i] and np.sum(workset[:i]) == pos:
                        workset[i] = False
                        break
        else:
            alpha = 1.0
            pos = -1
            for i in xrange(b.size):
                if not workset[i] and (A[i] * p_k)[0, 0] < 0:
                    now = np.abs((b[i] - A[i] * x) / (A[i] * p_k))[0, 0]
                    if now < alpha:
                        alpha = now
                        pos = i
            x += alpha * p_k
            if pos != -1:
                workset[pos] = True

    return x


def generate_data(train_size, test_size):
    mu1 = np.array([-1, -1])
    mu2 = np.array([1, 1])
    cov1 = cov2 = np.array([[0.5, -0.3], [-0.3, 0.5]])
    x1 = np.random.multivariate_normal(mu1, cov1, train_size + test_size)
    x2 = np.random.multivariate_normal(mu2, cov2, train_size + test_size)

    x_train = np.vstack((x1[:train_size], x2[:train_size]))
    x_test = np.vstack((x1[train_size:], x2[train_size:]))
    y_train = np.hstack((np.ones(train_size), -np.ones(train_size)))
    y_test = np.hstack((np.ones(test_size), -np.ones(test_size)))

    return x_train, y_train, x_test, y_test


def show(x_train, y_train, x_test, y_test, w):
    t = x_train[y_train == 1]
    plt.plot(t[:, 0], t[:, 1], 'ro', label='y = 1 in training data')
    t = x_train[y_train == -1]
    plt.plot(t[:, 0], t[:, 1], 'bo', label='y = -1 in training data')
    t = x_test[y_test == 1]
    plt.plot(t[:, 0], t[:, 1], 'rx', label='y = 1 in testing data')
    t = x_test[y_test == -1]
    plt.plot(t[:, 0], t[:, 1], 'bx', label='y = -1 in testing data')
    plt.ylim([-3.5, 3.5])
    plt.xlim([-3.5, 3.5])
    x = np.arange(-3.5, 3.5, 0.01)
    y = -w[1] / w[2] * x - w[0] / w[2]
    plt.plot(x, y, '-')
    # plt.legend()
    plt.show()


class SVM:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        n_features, n_samples = X.shape
        H = np.ones((n_features + 1, n_features + 1))
        H[0] = 0
        H[:, 0] = 0
        A = np.zeros((n_samples, n_features + 1))
        for i in xrange(n_samples):
            A[i] = np.hstack((y[i], y[i] * X[:, i]))
        self.w = quadprog(np.matrix(H), np.matlib.zeros((n_features + 1, 1)), A, np.matlib.ones((n_samples, 1)))

    def predict(self, X):
        return np.sign(self.w.T * X)


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = generate_data(70, 30)
    svm = SVM()
    svm.fit(x_train.T, y_train)
    print svm.w
    show(x_train, y_train, x_test, y_test, np.array(svm.w).T[0])
