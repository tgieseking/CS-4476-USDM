import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y)
    return X_train, X_test, y_train, y_test

def scale_train_test(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def split_seed_active(X, y):
    labels = np.unique(y)
    seed_indices = np.asarray([])
    for label in labels:
        label_indices = np.where(y == label)[0]
        if label_indices.size < 3:
            seed_indices = np.append(seed_indices, label_indices)
        else:
            label_seed_indices = label_indices[np.random.choice(label_indices.size, 3, replace=False)]
            seed_indices = np.append(seed_indices, label_seed_indices)
    seed_indices = seed_indices.astype(int)
    X_seed = X[seed_indices]
    y_seed = y[seed_indices]
    X_active = np.delete(X, seed_indices, axis=0)
    y_active = np.delete(y, seed_indices)
    return X_seed, X_active, y_seed, y_active
