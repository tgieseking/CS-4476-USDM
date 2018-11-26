import numpy as np
import pandas as pd

from subprocess import check_output

import pickle
import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from visualization import showConfusion

def train_svm(X_train, y_train):
    C_vals = [0.01, 0.1, 1., 10., 100.]
    gamma_vals = [0.00001, 0.0001, 0.001, 0.01, 0.1]


    try:
        splitter = StratifiedKFold(n_splits=5, shuffle=True)
        split_gen = splitter.split(X_train, y_train)
        splits = [split for split in split_gen]
    except:
        # This is because we can't do stratified k-fold if there are less per instance than folds
        splitter = KFold(n_splits=5, shuffle=True)
        split_gen = splitter.split(X_train, y_train)
        splits = [split for split in split_gen]

    scores = np.zeros((5, 5))

    for i, C in enumerate(C_vals):
        for j, gamma in enumerate(gamma_vals):
            print(C, gamma)
            trial_scores = []
            for train_inds, val_inds in splits:
                svm = SVC(C=C, gamma=gamma, kernel="rbf")
                svm.fit(X_train[train_inds], y_train[train_inds])
                y_pred = svm.predict(X_train[val_inds])
                trial_scores.append(f1_score(y_train[val_inds], y_pred, average = "micro"))
            scores[i, j] = np.average(trial_scores)
            print(scores)

    imax, jmax = np.unravel_index(np.argmax(scores), scores.shape)

    svm = SVC(C=C_vals[imax], gamma=gamma_vals[jmax], kernel="rbf")
    svm.fit(X_train, y_train)
    return svm, scores

def train_svm_parallel(X_train, y_train, use_stratified):
    param_grid = {"C": [0.01, 0.1, 1., 10., 100.], "gamma": [0.00001, 0.0001, 0.001, 0.01, 0.1]}

    splitter = KFold(n_splits=5, shuffle=True)

    grid_search = GridSearchCV(SVC(), param_grid, scoring="f1_micro", n_jobs=-1, cv=splitter, verbose=1)
    grid_search.fit(X_train, y_train)

    params = grid_search.best_params_
    print(params)

    svm = SVC(C=params["C"], gamma=params["gamma"], kernel="rbf")
    svm.fit(X_train, y_train)
    return svm, params, grid_search.cv_results_


def svm_train(X_train, X_test, y_train, y_test, target_names=[]):
    # digits = datasets.load_digits()

    # n_samples = len(digits['images'])
    # data_images = digits['images'].reshape((n_samples, -1))
    # target_names = digits['target_names']

    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(data_images, digits['target'])
    # print 'Training data and target sizes: \n{}, {}'.format(X_train.shape, y_train.shape)
    # print 'Test data and target sizes: \n{}, {}'.format(X_test.shape, y_test.shape)

    classifier = svm.SVC(gamma=0.001, kernel='rbf')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    if len(target_names) > 0:
        report = metrics.classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        pickle.dump(report, open("svm_output.npy", 'wb'))
        report = metrics.classification_report(y_test, y_pred, target_names=target_names)
        print report
    else:
        report = metrics.classification_report(y_test, y_pred, output_dict=True)
        pickle.dump(report, open("svm_output.npy", 'wb'))
        report = metrics.classification_report(y_test, y_pred)
        print report


    # classification_report_csv(report)
    # showConfusion(y_test, y_pred, target_names)

def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report.csv', index = False)
