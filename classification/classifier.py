import numpy as np
import pandas as pd

from subprocess import check_output

import pickle
import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics

from visualization import showConfusion

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


