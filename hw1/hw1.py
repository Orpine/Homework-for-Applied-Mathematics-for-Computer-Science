import theano
from theano import tensor as T
from theano import shared
import numpy as np
from matplotlib import pyplot as plt

curve_x = np.linspace(0, 1, 1000)


def plot_samples(n):
    sample_x = np.linspace(0, 1, n)
    sample_y = np.sin(sample_x * 2 * np.pi) + np.random.normal(0, 0.1, n)
    plt.plot(sample_x, sample_y, 'o')
    plt.plot(curve_x, np.sin(curve_x * 2 * np.pi), label='original curve')
    return sample_x, sample_y

def plot_poly(sample_x, sample_y, degree, regularization, label=None):
    theano.config.floatX = 'float64'
    X = T.matrix()
    y = T.vector()
    w = T.dot(T.dot(T.nlinalg.matrix_inverse(T.dot(X.T, X) + regularization * T.eye(degree + 1)), X.T), y)
    # loss = (T.sqr(T.dot(w, X) - y)).mean() + regularization * T.sum(T.sqr(w))
    # updates = [[w, w - 0.05 * T.grad(loss, w)]]
    # f = theano.function([X, y], [w], updates=updates, allow_input_downcast=True)
    f = theano.function([X, y], w, allow_input_downcast=True)
    train_X = np.ndarray((sample_x.size, degree + 1))
    test_X = np.ndarray((curve_x.size, degree + 1))
    for i in xrange(degree + 1):
        train_X[:, degree - i] = sample_x ** i
        test_X[:, degree - i] = curve_x ** i
    # for i in xrange(10000):
        # w = f(train_X, sample_y)
    w = f(train_X, sample_y)
    plt.plot(curve_x, np.dot(test_X, w), label=label)



if __name__ == "__main__":
    plt.figure(figsize=(18, 10))

    plt.subplot(221)
    plt.axis([-0.05, 1.05, -1.2, 1.2])
    sample_x, sample_y = plot_samples(10)
    plot_poly(sample_x, sample_y, 3, 0, label='M = 3')
    plot_poly(sample_x, sample_y, 9, 0, label='M = 9')
    plt.legend()

    plt.subplot(222)
    plt.axis([-0.05, 1.05, -1.2, 1.2])
    sample_x, sample_y = plot_samples(15)
    plot_poly(sample_x, sample_y, 9, 0, label='N = 15')
    plt.legend()

    plt.subplot(223)
    plt.axis([-0.05, 1.05, -1.2, 1.2])
    sample_x, sample_y = plot_samples(100)
    plot_poly(sample_x, sample_y, 9, 0, label='N = 100')
    plt.legend()

    plt.subplot(224)
    plt.axis([-0.05, 1.05, -1.2, 1.2])
    sample_x, sample_y = plot_samples(10)
    plot_poly(sample_x, sample_y, 9, np.exp(-18), label='ln$\lambda=-18$')
    plt.legend()
    plt.savefig('hw1.pdf')
    plt.show()
    # wait(100)