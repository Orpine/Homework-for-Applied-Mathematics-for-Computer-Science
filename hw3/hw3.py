import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    mean1 = [3, 0]
    mean2 = [-1, -3]
    mean3 = [4, 5]
    sigma1 = np.diag((1, 4))
    sigma2 = np.diag((4, 3))
    sigma3 = np.array([[2, 1], [1, 2]])
    # mean1 = [3, 0]
    # mean2 = [4, 5]
    # mean3 = [-5, -6]
    # sigma1 = np.eye(2)
    # sigma2 = np.eye(2)
    # sigma3 = np.eye(2)
    # sigma1 = np.diag((0.1, 0.2))
    # sigma2 = np.diag((0.3, 0.4))
    # sigma3 = np.array([[0.4, 0.3], [0.3, 0.4]])
    
    norm1 = np.random.multivariate_normal(mean1, sigma1, 300)
    norm2 = np.random.multivariate_normal(mean2, sigma2, 300)
    norm3 = np.random.multivariate_normal(mean3, sigma3, 300)
    x1, y1 = norm1.T
    x2, y2 = norm2.T
    x3, y3 = norm3.T
    data = np.vstack((norm1, norm2, norm3))
    np.save('data.npy', data)
    
    plt.plot(x1, y1, '.')
    plt.plot(x2, y2, '.')
    plt.plot(x3, y3, '.')
    plt.savefig('data.png')
    return data
    

class mog:
    def __init__(self, feas, dim, k, reg):
        self.feas = feas
        self.dim = dim
        self.k = k
        self.reg = reg
    
    def fit(self):
        center_idx = np.random.choice(self.feas.shape[0], self.k)
        self.centers = self.feas[center_idx, :]
        
        dis = np.zeros((self.feas.shape[0], self.k))
        for i in xrange(self.k):
            dis[:, i] = np.sum((self.feas - self.centers[i]) ** 2, axis=1)
        self.label = dis.argmin(axis=1)
        
        self.mean = self.centers
        self.sigma = np.zeros((self.feas.shape[1], self.feas.shape[1], self.k))
        self.prior = np.zeros((self.k,))
        for i in xrange(self.k):
            clus = self.feas[self.label == i]
            self.sigma[:, :, i] = np.cov(clus.T) + np.eye(self.feas.shape[1]) * self.reg
            self.prior[i] = 1. * clus.shape[0] / self.feas.shape[0]
            
        prob = self._EM()
        self.label = prob.argmax(axis=1)
        
        
    def _EM(self):
        
        pre_likelihood = np.inf
        while True:
            prob = self._prob()
            exp = prob * self.prior
            exp /= exp.sum(axis=1).reshape(exp.shape[0], 1)
            
            sum_k = exp.sum(axis=0)
            self.prior = sum_k / self.feas.shape[0]
            self.mean = np.diag(1. / sum_k).dot(exp.T).dot(self.feas)
            
            for i in xrange(self.k):
                shift = self.feas - self.mean[i, :]
                self.sigma[:, :, i] = shift.T.dot(np.diag(exp[:, i])).dot(shift) / sum_k[i] + np.eye(self.feas.shape[1]) * self.reg
                
            new_likelihood = np.log(prob * self.prior.T).sum()
            
            if np.abs(new_likelihood - pre_likelihood) < 1e-10:
                break
            pre_likelihood = new_likelihood
            
        return prob
        
        
    def _prob(self):
        prob = np.zeros((self.feas.shape[0], self.k))
        for i in xrange(self.k):
            shift = np.mat(self.feas - self.mean[i, :])
            exp_item = np.diag(shift * np.mat(self.sigma[:, :, i]).I * shift.T)
            coef = (2 * np.pi) ** (-self.feas.shape[1] / 2.) / np.sqrt(np.linalg.det(self.sigma[:, :, i]))
            prob[:, i] = coef * np.exp(-0.5 * exp_item)#.reshape((self.feas.shape[0]))
        return prob
    
    def show(self):
        plt.figure()
        for i in xrange(self.k):
            norm = self.feas[self.label == i]
            plt.plot(norm[:, 0], norm[:, 1], '.')
        # plt.show()

if __name__ == "__main__":
    feas = generate_data()
    gmm = mog(feas, 2, 3, 0.2)
    gmm.fit()
    gmm.show()
    gmm2 = mog(feas, 2, 3, 0)
    gmm2.fit()
    gmm2.show()    
    plt.show()