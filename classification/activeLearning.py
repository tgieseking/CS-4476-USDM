import sys
sys.path.append("./active_learning_master")
from sampling_methods.kcenter_greedy import kCenterGreedy
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from numpy.linalg import norm, inv, solve
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from scipy.optimize import minimize

def get_training_set(X_seed, y_seed, X_active, y_active, num_select, select_func):
    selected_indices = select_func(X_seed, y_seed, X_active, y_active, num_select)
    X_train = np.concatenate([X_seed, X_active[selected_indices]], axis=0)
    y_train = np.concatenate([y_seed, y_active[selected_indices]], axis=0)
    return X_train, y_train

def only_seed(X_seed, y_seed, X_active, y_active, num_select):
    return []

def all_active(X_seed, y_seed, X_active, y_active, num_select):
    return np.arange(y_active.size)

def uniform_random(X_seed, y_seed, X_active, y_active, num_select):
    N, _ = X_active.shape
    return np.random.choice(N, num_select, replace = False)

def margin(X_seed, y_seed, X_active, y_active, num_select):
    C = [1., 10., 100., 1000., 10000.]
    gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1]

    param_grid = {"C": [0.01, 0.1, 1., 10., 100.], "gamma": [0.00001, 0.0001, 0.001, 0.01, 0.1]}

    splitter = KFold(n_splits=5, shuffle=True)

    grid_search = GridSearchCV(SVC(), param_grid, scoring="f1_micro", n_jobs=-1, cv=splitter, verbose=1)
    grid_search.fit(X_seed, y_seed)

    params = grid_search.best_params_
    print(params)

    svm = SVC(C=params["C"], gamma=params["gamma"], kernel="rbf", probability = True)
    svm.fit(X_seed, y_seed)

    probs = svm.predict_proba(X_active)
    sorted_probs = np.sort(probs, axis = 1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    margins_sorted = np.argsort(margins)
    return margins_sorted[:num_select]

def k_center_greedy(X_seed, y_seed, X_active, y_active, num_select):
    num_seed, _ = X_seed.shape
    X_all = np.concatenate([X_seed, X_active], axis=0)
    y_all = np.concatenate([y_seed, y_active], axis=0)

    kcg = kCenterGreedy(X_all, y_all, 11)
    inds = kcg.select_batch_(None, np.arange(num_seed), num_select)
    return np.asarray(inds) - num_seed

def usdm(X_seed, y_seed, X_active, y_active, num_select):
    # Data processing
    seed_size, rows = X_seed.shape
    active_size, _ = X_active.shape
    N = seed_size + active_size
    X = np.concatenate((X_seed, X_active))
    classes = int(1 + np.max(y_active))

    # USDM parameters
    k = 5
    r = 1
    sigma = 1.0 * rows
    p_rand = 0.0001
    C = 100

    print("test1")
    neighbors = np.asarray(kneighbors_graph(X, k, include_self = False).todense())
    print("test2")
    dists = squareform(pdist(X))

    bneb = np.maximum(neighbors, neighbors.T)
    weights = dists * bneb
    sigma = np.average(weights[weights > 0])
    W = np.exp(-np.square(dists / sigma)) * neighbors
    W_norm = np.sum(W,axis=1, keepdims=True)
    Q = W / W_norm
    Q = (1 - p_rand) * Q + p_rand * np.ones(Q.shape) / Q.shape[1]
    Q_pp = Q[seed_size:,seed_size:]
    Q_sp = Q[seed_size:,:seed_size]
    Y = np.zeros((seed_size, classes))
    Y[np.arange(seed_size), y_seed] = 1

    print("test3")
    F = np.dot(np.dot(inv(np.eye(active_size) - Q_pp),Q_sp), Y)
    print("test4")
    b = np.concatenate((np.zeros(seed_size), np.sum(F * np.log(F), axis = 1)))
    a = b / np.abs(np.log(1.0/classes))
    print("step 1")

    sigma_K = np.average(dists)
    K = -np.square(dists / sigma_K)

    r = np.average(K) / np.average(a) / 2
    a *= r

    init = np.ones(N) / N
    fun = lambda f: np.dot(a, f) + np.dot(np.dot(K, f), f) / 2 + C * (np.square(np.sum(f) - 1)) / 2
    jac = lambda f: np.dot(K, f) + a + C * (np.sum(f) - 1) * np.ones(N)
    bounds = [(0,1.0/(num_select + seed_size))] * N
    result = minimize(fun, init, method="L-BFGS-B", jac=jac, bounds=bounds)
    active_scores = result.x[seed_size:]
    best_indices = np.argsort(active_scores)[-num_select:]
    print("finished")

    return best_indices
