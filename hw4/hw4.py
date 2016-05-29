import numdifftools as nd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# opt_function = lambda x: 1./4 * x[0] ** 4 - 1/3. * x[0] ** 3 + 1/2. * x[0] ** 2 - x[0] + 1
opt_function = lambda x: np.sin(x[0]) + np.cos(x[1])
grad_calculator = nd.Gradient(opt_function)
hessian_calculator = nd.Hessian(opt_function)

def grad(x):
    return grad_calculator(x)

def hessian(x):
    return hessian_calculator(x)

def Levenberg_Marquardt(x, max_iter=200, tol=1e-8):
    x = np.array(x)
    x_seq, y_seq, q_seq = [x], [opt_function(x)], [opt_function(x)]
    mu = 1e-3
    
    for iter in xrange(max_iter):
        g = grad(x)
        h = hessian(x)
        
        # print g
        # print (g ** 2).sum()
        if (g ** 2).sum() < tol:
            break
        
        while True:
            try:
                np.linalg.cholesky(h + mu * np.eye(h.shape[0]))
                break
            except:
                mu *= 4
        s = np.linalg.solve(h + mu * np.eye(h.shape[0]), -g)
        x_seq.append(x + s)
        y_seq.append(opt_function(x + s))
        q = y_seq[-1] + g.dot(s) + 0.5 * s.dot(h).dot(s.T)
        q_seq.append(q)
        # r = 1
        r = (y_seq[-1] - y_seq[-2]) / (q_seq[-1] - q_seq[-2])
        # print r
        
        if r < 0.25:
            mu *= 4
        elif r > 0.75:
            mu /= 2
        if r > 0:
            x = x + s
    # print q_seq
    # print y_seq
    return np.array(x_seq)


if __name__ == '__main__':
    x_sequence = Levenberg_Marquardt([2, 6])

    print x_sequence.shape
    print x_sequence
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(0, 2 * np.pi, 0.02)
    Y = np.arange(0, 2 * np.pi, 0.02)
    X, Y = np.meshgrid(X, Y)

    ax.plot_wireframe(X, Y, opt_function([X, Y]), rstride=10, cstride=10, label='f(X) = $sin(x_0) + cos(x_1)$')
    ax.plot3D(x_sequence[:, 0], x_sequence[:, 1], opt_function([x_sequence[:, 0], x_sequence[:, 1]]), 'r^-', label='X iteration')
    # plt.plot(t, map(opt_function, map(lambda x: [x], np.arange(-100.0, 100.0, 0.02))), 'b--', label='$x^4 + x^3 + x^2 + 1$')
    # plt.plot(x_sequence, map(opt_function, map(lambda x: [x], x_sequence)), 'r^-', label='x iteration')
    ax.legend()
    plt.grid(True)
    plt.show()

    
