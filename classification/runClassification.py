import numpy as np
from activeLearning import get_training_set, only_seed, all_active, uniform_random, margin, k_center_greedy, usdm
from classifier import train_svm, train_svm_parallel
from sklearn.metrics import f1_score
from scipy.io import savemat
import time
import sys

dataset = sys.argv[1]
selectName = sys.argv[2]
select_funcs = {"seed": only_seed, "all": all_active, "uniform": uniform_random, "margin": margin, "kcenter": k_center_greedy, "usdm": usdm}
select_func = select_funcs[selectName]


X_seed = np.load("data/%s_X_seed.npy" % dataset)
y_seed = np.load("data/%s_y_seed.npy" % dataset)
X_active = np.load("data/%s_X_active.npy" % dataset)
y_active = np.load("data/%s_y_active.npy" % dataset)
X_test = np.load("data/%s_X_test.npy" % dataset)
y_test = np.load("data/%s_y_test.npy" % dataset)

num_classes = 1 + np.max(y_active)
num_select = int(10 * num_classes)
X_train, y_train = get_training_set(X_seed, y_seed, X_active, y_active, num_select, select_func)

start = time.time()
svm, params, cv_results = train_svm_parallel(X_train, y_train, selectName != "seed")
y_pred = svm.predict(X_test)
score = f1_score(y_test, y_pred, average="micro")
print("f1 score: " + str(score))
runtime = str(time.time() - start)
print("time: " + runtime)

results = {"score": score, "y_pred": y_pred, "time": runtime, "params": params, "cv_results": cv_results}
savemat("results/%s_%s_results" % (dataset, selectName), results)
