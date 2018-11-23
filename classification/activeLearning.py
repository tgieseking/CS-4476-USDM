import numpy as np

def get_training_set(X_seed, y_seed, X_active, y_active, num_select, select_func):
    selected_indices = select_func(X_seed, y_seed, X_active, num_select)
    X_train = np.concatenate([X_seed, X_active[selected_indices]], axis=0)
    y_train = np.concatenate([y_seed, y_active[selected_indices]], axis=0)
    return X_train, y_train

def uniform_random(X_seed, y_seed, X_active, num_select):
    N, _ = X_active.shape
    return np.random.choice(N, num_select, replace = False)
