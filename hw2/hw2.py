import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    feas = []
    with open('optdigits.tra', 'r') as f:
        for l in f:
            if l[-2] == '3':
                l = l[:-3].strip().split(',')
                feas.append([float(x) for x in l])
    # with open('optdigits.tes', 'r') as f:
    #     for l in f:
    #         if l[-2] == '3':
    #             l = l[:-3].strip().split(',')
    #             feas.append([float(x) for x in l])
    feas = np.array(feas) # N x d
    
    m = feas.mean(axis = 0)
    feas = (feas - m)
    # U, S, VT = np.linalg.svd(np.dot(feas, feas.T))
    # res = np.dot(U[:, :2].T, feas)
    U, S, VT = np.linalg.svd(feas, full_matrices = False)
    # res = np.dot(feas, VT[0:2, :].T)
    # res = np.dot(feas, U[0:2].T)
    res = np.dot(U[:, 0:2], np.diag(S[0:2]))
        
    location = []
    for i in xrange(5):
        for j in xrange(5):
            tagX = -20 + 10 * i
            tagY = 20 - 10 * j
            now = np.inf
            idx = 0
            for k in xrange(res.shape[0]):
                dis = (res[k, 0] - tagX) ** 2 + (res[k, 1] - tagY) ** 2
                if dis < now:
                    now = dis
                    idx = k
            location.append(idx)
    
    plt.figure(figsize=(10, 6))
    plt.subplot2grid((1, 3), (0, 0), colspan=2)
    x = res[:, 0]
    y = res[:, 1]
    
    plt.plot(x, y, '.g')
    locationX = [res[idx, 0] for idx in location]
    locationY = [res[idx, 1] for idx in location]
    
    plt.plot(locationX, locationY, 'or')
    plt.grid(color='grey')
    plt.xlabel('First PC')
    plt.ylabel('Second PC')
    
    plt.subplot2grid((1, 3), (0, 2))
    img = np.zeros((40, 40))
    for i in xrange(len(location)):
        row = (i % 5) * 8
        col = (i / 5) * 8
        three = np.array(feas[location[i], :] + m)
        three = three.reshape((8, 8))
        img[row:row+8, col:col+8] = three
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
    
    
    