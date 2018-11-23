import numpy as np
from activeLearning import get_training_set, uniform_random
from classifier import train_svm
from sklearn.metrics import f1_score
import time

X_seed = np.load("data/COIL_X_seed.npy")
y_seed = np.load("data/COIL_y_seed.npy")
X_active = np.load("data/COIL_X_active.npy")
y_active = np.load("data/COIL_y_active.npy")
X_test = np.load("data/COIL_X_test.npy")
y_test = np.load("data/COIL_y_test.npy")

active_size, _ = X_active.shape
num_select = int(0.4 * active_size)
X_train, y_train = get_training_set(X_seed, y_seed, X_active, y_active, num_select, uniform_random)

start = time.time()
svm = train_svm(X_train, y_train)
y_pred = svm.predict(X_test)
score = f1_score(y_test, y_pred, average="micro")
print("f1 score: " + str(score))
print("time: " + str(time.time() - start))

import pdb; pdb.set_trace()
