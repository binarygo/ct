import os
import sys
import time
import pickle
import numpy as np
import csv
import pandas as pd
from skimage import measure

import xgboost as xgb

from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import Normalizer

import util
import luna_preprocess
import kagl_preprocess
import kagl_test_unet5
import kagl_feature_util
import kagl_feature_set1


def prepare_data(names, meta_patient, feature_set):
    feature_np = []
    label_np = []
    for name in names:
        label_np.append(meta_patient.labels[name])
        feature_np.append(feature_set[name])
    feature_np = np.asarray(feature_np, dtype=np.float)
    label_np = np.asarray(label_np, dtype=np.float)
    
    try:
        print 'feature.shape = ', feature_np.shape
        print 'pos = %d, %f'%(
            np.sum(label_np),
            np.sum(label_np) * 1.0 / len(label_np))
        print 'neg = %d, %f'%(
            len(label_np) - np.sum(label_np),
            1.0 - np.sum(label_np) * 1.0 / len(label_np))
    except:
        pass
    
    col_idxes = [14,12,10,54,89,13,91,15,1,23,2,26,28,59,22,75,32,77,61,16,17,25,82,103,63,114,42,19,68,56]

    feature_np = feature_np[:, col_idxes]

    return feature_np, label_np


stage = 'stage1'
meta_patient = kagl_preprocess.MetaPatient(stage)

train_names = []
test_names = []
for name, label in meta_patient.labels.iteritems():
    if label is not None:
        train_names.append(name)
    else:
        test_names.append(name)

feature_set = np.load('kagl_output_feature_set1.npy').item()

train_feature_np, train_label_np = prepare_data(
    train_names, meta_patient, feature_set)

test_feature_np, _ = prepare_data(
    test_names, meta_patient, feature_set)

X = train_feature_np
Y = train_label_np

#normalizer = Normalizer(copy=False)
normalizer = None
if normalizer is None:
    X_scaled = X
else:
    X_scaled = normalizer.fit_transform(X)    


print("Predicting all positive")
y_pred = np.ones(Y.shape)
print(classification_report(
      Y, y_pred, target_names=["No Cancer", "Cancer"]))
print("logloss", kagl_feature_util.logloss(Y, y_pred))

# No Cancer
print("Predicting all negative")
y_pred = Y * 0
print(classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"]))
print("logloss", kagl_feature_util.logloss(Y, y_pred))


skf = StratifiedKFold(n_splits=10)
kf = skf.split(X, Y)

y_pred_rf = Y * 0.0
y_pred_svc = Y * 0.0
y_pred_xgb = Y * 0.0
for train, valid in kf:
    X_train, X_valid, y_train, y_valid = (
        X_scaled[train, :], X_scaled[valid, :], Y[train], Y[valid])
    clf_rf = RF(n_estimators=10000, criterion='entropy', max_depth=None, n_jobs=3)
    clf_rf.fit(X_train, y_train)
    y_pred_tmp = clf_rf.predict_proba(X_valid)[:,1]
    y_pred_rf[valid] = y_pred_tmp
    
    clf_svc = SVC(
        class_weight={0: 2},
        probability=True,
        shrinking=True,
        C=0.01,
        gamma=1000)
    clf_svc.fit(X_train, y_train)
    y_pred_tmp = clf_svc.predict_proba(X_valid)[:,1]
    y_pred_svc[valid] = y_pred_tmp

    clf_xgb = xgb.XGBClassifier(
        max_depth=100,
        learning_rate=0.0001,
        n_estimators=256,
        objective="binary:logistic")
    clf_xgb.fit(X_train, y_train)
    y_pred_tmp = clf_xgb.predict_proba(X_valid)[:,1]
    y_pred_xgb[valid] = y_pred_tmp
    
print 'Random forest'
print(classification_report(
        Y, y_pred_rf > 0.5, target_names=["No Cancer", "Cancer"]))
print("logloss", kagl_feature_util.logloss(Y, y_pred_rf))

print 'SVM'
print(classification_report(
        Y, y_pred_svc > 0.5, target_names=["No Cancer", "Cancer"]))
print("logloss", kagl_feature_util.logloss(Y, y_pred_svc))

print 'XGB'
print(classification_report(
        Y, y_pred_xgb > 0.5, target_names=["No Cancer", "Cancer"]))
print("logloss", kagl_feature_util.logloss(Y, y_pred_xgb))


if normalizer is None:
    X_test = test_feature_np
else:
    X_test = normalizer.transform(X_test)

print 'Write submission: Random forest'
y_test = clf_rf.predict_proba(X_test)[:,1]
kagl_feature_util.write_submission_file(test_names, y_test, 'rf')

print 'Write submission: SVM'
y_test = clf_svc.predict_proba(X_test)[:,1]
kagl_feature_util.write_submission_file(test_names, y_test, 'svm')

print 'Write submission: XGB'
y_test = clf_xgb.predict_proba(X_test)[:,1]
kagl_feature_util.write_submission_file(test_names, y_test, 'xgb')
