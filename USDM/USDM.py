import numpy as np
from numpy.linalg import norm, inv, solve
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph

def select_batch(seed_x, seed_y, active_x, num_select, classes, iterations):
    seed_size, _ = seed_x.shape
    active_size, _ = active_x.shape
    N = seed_size + active_size
    X = np.concatenate((seed_x, active_x))
    Y = np.concatenate((seed_y, active_y))
    
    k = 5
    r = 1
    sigma = 1
    
    neighbors = np.asarray(kneighbors_graph(X, k, include_self = True).todense())
    dists = squareform(pdist(X))
    W = np.exp(-np.square(dists)) * neighbors
    W_norm = np.sum(W,axis=1, keepdims=True)
    Q = W / W_norm
    Q_pp = Q[seed_size:,seed_size:]
    Q_sp = Q[seed_size:,:seed_size]
    Y = np.zeros((seed_size, classes))
    Y[np.arange(seed_size), seed_y] = 1
    F = np.dot(np.dot(inv(np.eye(active_size) - Q_pp),Q_sp), Y)
    b = np.concatenate((np.zeros(seed_size), np.sum(F * np.log(F), axis = 1)))
    a = b * r / np.abs(np.log(1.0/classes))
    
    K = -np.square(dists / sigma)
    
    rho = 1.1
    f = np.ones(N) / N
    v = np.copy(f)
    l1 = 0
    l2 = np.zeros(N)
    mu = 0.01
    
    for i in range(iterations):
        A = K + (mu * np.eye(N)) + (mu * np.ones((N,N)))
        e = (mu * v) + (mu * np.ones(N)) - (l1 * np.ones(N)) - l2 - a
        f_hat = solve(A, e)
        v = np.maximum(f_hat + l2 / mu, 0)
        l1 = l1 + mu * (np.sum(f_hat) - 1)
        l2 = l2 + mu * (f_hat - v)
        mu = rho * mu
        
#         opt = np.dot(f_hat, a) + np.dot(np.dot(f_hat.T, A), f_hat) / 2
#         print(opt, np.sum(f_hat))
    

    f_active = f_hat[seed_size:]
#     print(f_active)
    return np.argpartition(f_active, f_active.size - num_select)[f_active.size - num_select:]

