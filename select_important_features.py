#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File    ：master_thesis -> select_important_features
@IDE    ：PyCharm
@Author  : Xue-Qiang Fan
@Date   ：2022/3/25 12:11
=================================================='''
import numpy as np
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, RFE
from I_DNAN6mAplus import save
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, \
    RandomForestClassifier, \
    AdaBoostClassifier, \
    GradientBoostingClassifier, \
    ExtraTreesClassifier

def accordBestIndexExtractFeature(best_index, X):
    scale = StandardScaler()
    X = scale.fit_transform(X)
    index = pd.read_csv(best_index, header=None)
    index = np.squeeze(index.values, 0)
    return X[:, index]

def accordProbConvertValue(pred_prob):
    value = []
    for i in range(pred_prob.shape[0]):
        if pred_prob[i] < 0.5:
            pred = 0
            value.append(pred)
        elif 0.5 <= pred_prob[i]:
            pred = 1
            value.append(pred)
    return value

def calculate_TPR_FPR(predict, label):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(label.shape[0]):
        pre, lab = predict[i], label[i]

        if lab == 0:
            if pre == 0:
                TN += 1
            elif pre == 1:
                FP += 1
        elif lab == 1:
            if pre == 0:
                FN += 1
            elif pre == 1:
                TP += 1
    TPR = TP/(TP+FN)
    FPR = 1-TN/(TN+FP)
    return TPR, FPR

def calculateFiveClassifierVoting(stack_prob, label, threshold, votes):
    L, N = stack_prob.shape
    stack_value = np.zeros([L, N])
    for i in range(L):
        for j in range(N):
            if stack_prob[i, j] < threshold:
                stack_value[i, j] = 0
            elif stack_prob[i, j] >= threshold:
                stack_value[i, j] = 1
    stack_value = np.sum(stack_value, 1)
    y_artificial = np.zeros(L)
    for i in range(L):
        if stack_value[i] >= votes:
            y_artificial[i] = 1
        else:
            y_artificial[i] = 0
    TPR, FPR = calculate_TPR_FPR(y_artificial, label)
    return TPR, FPR, threshold, y_artificial


def selectKImportanceFeatures(X, Y):
    scaler = StandardScaler()
    x = scaler.fit_transform(X)
    # model = LinearSVC(max_iter=10)
    # model = RandomForestClassifier(n_estimators=5)
    # model = LogisticRegression(max_iter=3)
    # model = AdaBoostClassifier(n_estimators=5)
    model = XGBClassifier(object='binary:logistic', verbosity=0, eval_metric=['logloss', 'auc', 'error'],
                          n_estimators=500, subsample=0.8, colsample_btree=0.8, colsample_bytree=0.8, learning_rate=0.3)
    # refcv = RFECV(estimator=model, step=200, min_features_to_select=100, cv=StratifiedKFold(5))  # 挑选200重要特征
    ref = RFE(estimator=model, n_features_to_select=80, step=10)  # 挑选200重要特征
    # refcv = RFECV(model, 100)
    ref.fit(x, Y)
    # print("Num Features: %d" % refcv.n_features_)
    # print("Feature Ranking: %s" % refcv.ranking_)
    best_Index = [i for i, x in enumerate(ref.ranking_) if x == 1]
    print(len(best_Index))
    save.saveBestK(best_Index)

    IF = ref.transform(X)

    score = cross_val_score(model, IF, Y, cv=5).mean()
    print("Accuracy: %0.4f(+/-%0.4f)" % (score.mean(), score.std() * 5))

    return IF
