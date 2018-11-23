import sys
sys.path.append("./active_learning_master")
from sampling_methods.kcenter_greedy import kCenterGreedy
import numpy as np

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
# TODO
    return None

def k_center_greedy(X_seed, y_seed, X_active, y_active, num_select):
    num_seed, _ = X_seed.shape
    X_all = np.concatenate([X_seed, X_active], axis=0)
    y_all = np.concatenate([y_seed, y_active], axis=0)

    kcg = kCenterGreedy(X_all, y_all, 11)
    inds = kcg.select_batch_(None, np.arange(num_seed), num_select)
    return np.asarray(inds) - num_seed

def usdm(X_seed, y_seed, X_active, y_active, num_select):
# TODO
    return None
