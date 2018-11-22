import numpy as np
import operator

from classifier import svm_train

X_active = np.load('datasets/COIL_X_active.npy')
X_test = np.load('datasets/COIL_X_test.npy')
y_active = np.load('datasets/COIL_y_active.npy')
y_test = np.load('datasets/COIL_y_test.npy')
X_seed = np.load('datasets/COIL_X_seed.npy')
y_seed = np.load('datasets/COIL_y_seed.npy')
# print X_train.shape
print X_test.shape
# print y_train.shape
print y_test.shape
# print X_seed.shape
# print y_seed.shape
X_train = np.concatenate((X_active, X_seed))
y_train = np.concatenate((y_active, y_seed))
print X_train.shape
print y_train.shape

labels = np.load('datasets/labels.npy')
target_names = sorted(labels, key=labels.get)

svm_train(X_seed, X_test, y_seed, y_test)