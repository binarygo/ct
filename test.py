# script to process

import os

import csv
import numpy as np
import scipy as sp

from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as xgb
from matplotlib import pyplot as plt

import sys
import csv
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage import morphology
from glob import glob

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import Normalizer


def read_patient_labels(csv_fname):
    with open(csv_fname) as f:
        reader = csv.reader(f, delimiter=',')
        reader.next()  # skip header
        return dict([(r[0], float(r[1])) for r in reader if len(r) == 2])


_DATA_DIR = '../KAGL16/stage1'
_LABELS_CSV = '../KAGL16/stage1_labels.csv'
_SAMPLE_CSV = '../KAGL16/stage1_sample_submission.csv'


patient_names = os.listdir(_DATA_DIR)
patient_labels = read_patient_labels(_LABELS_CSV)
test_patient_names = list(set(read_patient_labels(_SAMPLE_CSV).keys()))

print("[ground-truth patients, test-patients] = [%d, %d]"
    % (len(patient_labels), len(test_patient_names)))


def classify_data():
    # load feature data
    print 'load feature data'
    feature_map = np.load('./feature_map_thres_5.dat')
    #feature_map.keys()
    X_map = feature_map[()]

    # use patients with ground truth for training
    patients = patient_labels.keys()

    X = np.array([X_map[p] for p in patients])
    Y = np.array([patient_labels[p] for p in patients]).astype('float32')

    print Y
    print '[pos={}, neg={}, all={}]'.format(sum(Y), len(Y) - sum(Y), len(Y))

    print X.shape

    X = X.clip(min=1e-5)
    #ch2 = SelectKBest(chi2, k=20)
    #X = ch2.fit_transform(X, Y)

    print X.shape
    print X[0, :]

    normalizer = Normalizer(copy=False)
    X_scaled = normalizer.fit_transform(X)
    print X_scaled[0, :]

    # All Cancer
    print("Predicting all positive")
    y_pred = np.ones(Y.shape)
    print(classification_report(
        Y, y_pred, target_names=["No Cancer", "Cancer"]))
    print("logloss", logloss(Y, y_pred))

    # No Cancer
    print("Predicting all negative")
    y_pred = Y * 0
    print(
        classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"]))
    print("logloss", logloss(Y, y_pred))

    skf = StratifiedKFold(n_splits=5)
    kf = skf.split(X, Y)
    y_pred = Y * 0.5
    y_pred_svc = Y * 0.5
    y_pred_xgb = Y * 0.5
    for train, test in kf:
        X_train, X_test, y_train, y_test = \
            X_scaled[train, :], X_scaled[test, :], Y[train], Y[test]
        clf = RF(n_estimators=10000, n_jobs=3, criterion='entropy')
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict_proba(X_test)[:,1]

        #print y_train
        #print y_train[y_train>0]

        clf_svc = SVC(
            class_weight={0: 2},
            probability=True,
            shrinking=True,
            C=0.01,
            gamma=1000)
        clf_svc.fit(X_train, y_train)
        y_pred_svc[test] = clf_svc.predict_proba(X_test)[:,1]

        clf_xgb = xgb.XGBClassifier(objective="binary:logistic")
        clf_xgb.fit(X_train, y_train)
        y_pred_xgb[test] = clf_xgb.predict_proba(X_test)[:,1]

    print 'Random forest'
    print(classification_report(
        Y, y_pred > 0.5, target_names=["No Cancer", "Cancer"]))
    print("logloss", logloss(Y, y_pred))

    print 'SVM'
    print(classification_report(
        Y, y_pred_svc > 0.5, target_names=["No Cancer", "Cancer"]))
    print("logloss", logloss(Y, y_pred_svc))

    print 'XGB'
    print(classification_report(
        Y, y_pred_xgb > 0.5, target_names=["No Cancer", "Cancer"]))
    print("logloss", logloss(Y, y_pred_xgb))

    # ######################
    # Classify test data
    # ######################

    print("extract features for test patients")
    X_test_patient = \
        np.array([X_map[p] for p in test_patient_names])
    X_test_patient = normalizer.transform(X_test_patient)

    #print X_test_patient[0,:]

    clf_svc.fit(X_scaled, Y)
    test_patient_pred = clf_svc.predict_proba(X_test_patient)[:,1]
    #print(test_patient_pred)
    write_submission_file(test_patient_names, test_patient_pred, 'svm-2')

    clf.fit(X_scaled, Y)
    test_patient_pred = clf.predict_proba(X_test_patient)[:,1]
    #print(test_patient_pred)
    write_submission_file(test_patient_names, test_patient_pred, 'rf-2')


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act)
        * sp.log(sp.subtract(1, pred)))
    ll = ll * - 1.0 / len(act)
    return ll


def write_submission_file(patient_names, preds, file_name=''):
    with open('lcad-{}.csv'.format(file_name), 'w') as f:
        f.write('id,cancer\n')
        for i in range(len(patient_names)):
            f.write('{},{}\n'.format(patient_names[i], preds[i]))


if __name__ == '__main__':
    classify_data()
