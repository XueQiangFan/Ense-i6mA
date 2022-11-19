#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File    ：master_thesis -> runnClassifiers
@IDE    ：PyCharm
@Author  : Xue-Qiang Fan
@Date   ：2022/3/25 11:13
=================================================='''
# Avoiding warning
import warnings


def warn(*args, **kwargs): pass


warnings.warn = warn
from scipy import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from I_DNAN6mAplus.select_important_features import accordProbConvertValue, calculateFiveClassifierVoting
# _____________________________
# np.random.seed(seed=111)
# scikit-learn :
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
from I_DNAN6mA_D.main import main
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, \
    RandomForestClassifier, \
    AdaBoostClassifier, \
    GradientBoostingClassifier, \
    ExtraTreesClassifier

a = 0
b = 1
c = 2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Classifiers = [ExtraTreesClassifier(n_estimators=5), ]
# Names = ['ET']
Names = ['GB', 'SVM', 'ET', 'XGB', 'I-DNAN6mA', "Stack-Averaging", "Stack-Voting"]
Classifiers = [
    GradientBoostingClassifier(n_estimators=500),  # 8
    SVC(C=15.0, kernel='rbf', degree=3, probability=True),  # 9
    ExtraTreesClassifier(n_estimators=500),  # 11
    XGBClassifier(object='binary:logistic', verbosity=0, max_iter=500,
                  eval_metric=['logloss', 'auc', 'error'],
                  n_estimators=500, subsample=0.8, colsample_btree=0.8, colsample_bytree=0.8,
                  learning_rate=0.1),
    a,
    b,
    c,
]

F = open('A_test_evaluationResults.txt', 'w')

F.write('Evaluation Scale:' + '\n')
F.write('0.0% <=Accuracy<= 100.0%' + '\n')
F.write('0.0 <=auROC<= 1.0' + '\n')
F.write('0.0 <=auPR<= 1.0' + '\n')  # average_Precision
F.write('0.0 <=F1_Score<= 1.0' + '\n')
F.write('-1.0 <=MCC<= 1.0' + '\n')
F.write('0.0%<=Sensitivity<= 100.0%' + '\n')
F.write('0.0%<=Specificity<= 100.0%' + '\n')


def runClassifiers(args, X_train, Y_train, X_test, Y_test):
    # D = pd.read_csv(args.dataset, header=None)  # Using R
    # print('Before drop duplicates: {}'.format(D.shape))
    # D = D.drop_duplicates()  # Return : each row are unique value
    # print('After drop duplicates: {}\n'.format(D.shape))

    X = X_train
    y = Y_train

    from sklearn.preprocessing import StandardScaler
    scale = StandardScaler()
    X = scale.fit_transform(X)

    # D = pd.read_csv(args.test_dataset, header=None)  # Using R
    # print('Before drop duplicates: {}'.format(D.shape))
    # D = D.drop_duplicates()  # Return : each row are unique value
    # print('After drop duplicates: {}\n'.format(D.shape))

    X_test = X_test
    y_test = Y_test

    scale = StandardScaler()
    X_test = scale.fit_transform(X_test)

    F.write('\n' + '------------------------------------------' + '\n')
    # F.write('Using {} fold-cross validation results.\n'.format(args.nFCV))
    F.write('------------------------------------------' + '\n')

    Results = []  # compare algorithms

    from sklearn.metrics import accuracy_score, \
        log_loss, \
        classification_report, \
        confusion_matrix, \
        roc_auc_score, \
        average_precision_score, \
        auc, \
        roc_curve, f1_score, recall_score, matthews_corrcoef, auc

    # Step 05 : Spliting with 10-FCV :
    from sklearn.model_selection import StratifiedKFold

    # cv = StratifiedKFold(n_splits=args.nFCV, shuffle=True)

    stack_prob_2 = np.array([], float)
    stack_prob_3 = np.array([], float)
    stack_prob_4 = np.array([], float)
    stack_prob_5 = np.array([], float)
    # stack_value = []
    for classifier, name in zip(Classifiers, Names):

        accuray = []
        auROC = []
        avePrecision = []
        F1_Score = []
        AUC = []
        MCC = []
        Recall = []
        LogLoss = []

        mean_TPR = 0.0
        mean_FPR = np.linspace(0, 1, 100)
        CM = np.array([
            [0, 0],
            [0, 0],
        ], dtype=int)

        print('{} is done.'.format(classifier.__class__.__name__))
        F.write(classifier.__class__.__name__ + '\n\n')
        if name == "I-DNAN6mA":
            y_test, y_proba = main()

            stack_prob = np.expand_dims(np.append(stack_prob, y_proba), 1)
            print(stack_prob.shape)
            # stack_value.append(accordProbConvertValue(y_proba))

            FPR, TPR, _ = roc_curve(y_test, y_proba)
            print(interpolate.interp1d(FPR, TPR)(0.1))
            print(interpolate.interp1d(FPR, TPR)(0.2))
            mean_TPR += np.interp(mean_FPR, FPR, TPR)
            mean_TPR[0] = 0.0
            auROC.append(roc_auc_score(y_true=y_test, y_score=y_proba))
            mean_auc = auc(mean_FPR, mean_TPR)
            mean_TPR[-1] = 1.0
            plt.plot(
                mean_FPR,
                mean_TPR,
                linestyle='-',
                label='{} ({:0.3f})'.format(name, 0.965), lw=2.0)

        elif name == "Stack-Averaging":
            stack_prob_mean = np.mean(stack_prob, 1)
            y_proba = stack_prob_mean
            FPR, TPR, _ = roc_curve(y_test, y_proba, pos_label=1)  # TPR=TP/(TP+FN) FPR=FP/(FP+TN)
            print(interpolate.interp1d(FPR, TPR)(0.1))
            print(interpolate.interp1d(FPR, TPR)(0.2))
            mean_TPR += np.interp(mean_FPR, FPR, TPR)
            mean_TPR[0] = 0.0
            roc_auc = auc(FPR, TPR)

            y_artificial = np.array(accordProbConvertValue(y_proba), int)

            auROC.append(roc_auc_score(y_true=y_test, y_score=y_proba))
            accuray.append(accuracy_score(y_pred=y_artificial, y_true=y_test))
            avePrecision.append(average_precision_score(y_true=y_test, y_score=y_proba))  # auPR
            F1_Score.append(f1_score(y_true=y_test, y_pred=y_artificial))
            MCC.append(matthews_corrcoef(y_true=y_test, y_pred=y_artificial))
            Recall.append(recall_score(y_true=y_test, y_pred=y_artificial))
            AUC.append(roc_auc)
            CM += confusion_matrix(y_pred=y_artificial, y_true=y_test)

            print("{} --- over.".format(name))

            accuray = [_ * 100.0 for _ in accuray]
            Results.append(accuray)

            # mean_TPR /= cv.get_n_splits(X, y)
            mean_TPR[-1] = 1.0
            # print(mean_TPR)
            mean_auc = auc(mean_FPR, mean_TPR)
            plt.plot(
                mean_FPR,
                mean_TPR,
                linestyle='-',
                label='{} ({:0.3f})'.format(name, mean_auc), lw=2.0)

            F.write('Accuracy: {0:.4f}%\n'.format(np.mean(accuray)))
            # print('auROC: {0:.6f}'.format(np.mean(auROC)))
            F.write('auROC: {0:.4f}\n'.format(mean_auc))
            # F.write('AUC: {0:.4f}\n'.format( np.mean(AUC)))
            F.write('auPR: {0:.4f}\n'.format(np.mean(avePrecision)))  # average_Precision
            F.write('F1_Score: {0:.4f}\n'.format(np.mean(F1_Score)))
            F.write('MCC: {0:.4f}\n'.format(np.mean(MCC)))

            TN, FP, FN, TP = CM.ravel()
            F.write('Recall: {0:.4f}\n'.format(np.mean(Recall)))
            F.write('Sensitivity: {0:.4f}%\n'.format(((TP) / (TP + FN)) * 100.0))
            F.write('Specificity: {0:.4f}%\n'.format(((TN) / (TN + FP)) * 100.0))
            F.write('Confusion Matrix:\n')
            F.write(str(CM) + '\n')
            F.write('_______________________________________' + '\n')

        elif name == "Stack-Voting":
            TPR, FPR, thresholds = [], [], []
            for i in [0.999, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2,
                      0.15, 0.1, 0.05, 0.01, 0.001]:
                tpr, fpr, threshold, y_artificial = calculateFiveClassifierVoting(stack_prob, y_test, i)
                TPR.append(tpr)
                FPR.append(fpr)
                thresholds.append(i)
            TPR, FPR, thresholds = np.array(TPR), np.array(FPR), np.array(thresholds),
            print(TPR, FPR, thresholds)
            print(interpolate.interp1d(FPR, TPR)(0.1))
            print(interpolate.interp1d(FPR, TPR)(0.2))
            mean_TPR += np.interp(mean_FPR, FPR, TPR)
            mean_TPR[0] = 0.0
            roc_auc = auc(FPR, TPR)

            # auROC.append(roc_auc_score(y_true=y_test, y_score=y_proba))

            tpr, fpr, threshold, y_artificial = calculateFiveClassifierVoting(stack_prob, y_test, 0.5)
            accuray.append(accuracy_score(y_pred=y_artificial, y_true=y_test))
            # avePrecision.append(average_precision_score(y_true=y_test, y_score=y_proba))  # auPR
            F1_Score.append(f1_score(y_true=y_test, y_pred=y_artificial))
            MCC.append(matthews_corrcoef(y_true=y_test, y_pred=y_artificial))
            Recall.append(recall_score(y_true=y_test, y_pred=y_artificial))
            AUC.append(roc_auc)
            CM += confusion_matrix(y_pred=y_artificial, y_true=y_test)

            print("{} --- over.".format(name))

            accuray = [_ * 100.0 for _ in accuray]
            Results.append(accuray)

            # mean_TPR /= cv.get_n_splits(X, y)
            mean_TPR[-1] = 1.0
            # print(mean_TPR)
            mean_auc = auc(mean_FPR, mean_TPR)
            plt.plot(
                mean_FPR,
                mean_TPR,
                linestyle='-',
                label='{} ({:0.3f})'.format(name, mean_auc), lw=2.0)

            F.write('Accuracy: {0:.4f}%\n'.format(np.mean(accuray)))
            # print('auROC: {0:.6f}'.format(np.mean(auROC)))
            F.write('auROC: {0:.4f}\n'.format(mean_auc))
            # F.write('AUC: {0:.4f}\n'.format( np.mean(AUC)))
            # F.write('auPR: {0:.4f}\n'.format(np.mean(avePrecision)))  # average_Precision
            F.write('F1_Score: {0:.4f}\n'.format(np.mean(F1_Score)))
            F.write('MCC: {0:.4f}\n'.format(np.mean(MCC)))

            TN, FP, FN, TP = CM.ravel()
            F.write('Recall: {0:.4f}\n'.format(np.mean(Recall)))
            F.write('Sensitivity: {0:.4f}%\n'.format(((TP) / (TP + FN)) * 100.0))
            F.write('Specificity: {0:.4f}%\n'.format(((TN) / (TN + FP)) * 100.0))
            F.write('Confusion Matrix:\n')
            F.write(str(CM) + '\n')
            F.write('_______________________________________' + '\n')



        else:
            model = classifier
            model.fit(X, y)

            # if name == "SVM":
            #     with open('A_train_SVM_dumpModel.pkl', 'wb') as File:
            #         joblib.dump(model, File)
            # if name == "XGB":
            #     with open('A_train_XGB_dumpModel.pkl', 'wb') as File:
            #         joblib.dump(model, File)
            # if name == "ET":
            #     with open('A_train_ET_dumpModel.pkl', 'wb') as File:
            #         joblib.dump(model, File)
            # if name == "GB":
            #     with open('A_train_GB_dumpModel.pkl', 'wb') as File:
            #         joblib.dump(model, File)

            # print(model.predict(X_train))

            # Calculate ROC Curve and Area the Curve
            y_proba = model.predict_proba(X_test)[:, 1]
            y_proba = np.expand_dims(y_proba, 1)
            stack_prob = np.append(stack_prob, y_proba, axis=1)

            FPR, TPR, _ = roc_curve(y_test, y_proba, pos_label=1)  # TPR=TP/(TP+FN) FPR=FP/(FP+TN)
            # print(FPR.shape, TPR.shape, _.shape)
            print(interpolate.interp1d(FPR, TPR)(0.1))
            print(interpolate.interp1d(FPR, TPR)(0.2))
            mean_TPR += np.interp(mean_FPR, FPR, TPR)
            mean_TPR[0] = 0.0
            roc_auc = auc(FPR, TPR)

            y_artificial = model.predict(X_test)
            # print(y_artificial)

            auROC.append(roc_auc_score(y_true=y_test, y_score=y_proba))
            accuray.append(accuracy_score(y_pred=y_artificial, y_true=y_test))
            avePrecision.append(average_precision_score(y_true=y_test, y_score=y_proba))  # auPR
            F1_Score.append(f1_score(y_true=y_test, y_pred=y_artificial))
            MCC.append(matthews_corrcoef(y_true=y_test, y_pred=y_artificial))
            Recall.append(recall_score(y_true=y_test, y_pred=y_artificial))
            AUC.append(roc_auc)
            CM += confusion_matrix(y_pred=y_artificial, y_true=y_test)

            print("{} --- over.".format(name))

            accuray = [_ * 100.0 for _ in accuray]
            Results.append(accuray)

            # mean_TPR /= cv.get_n_splits(X, y)
            mean_TPR[-1] = 1.0
            # print(mean_TPR)
            mean_auc = auc(mean_FPR, mean_TPR)
            plt.plot(
                mean_FPR,
                mean_TPR,
                linestyle='-',
                label='{} ({:0.3f})'.format(name, mean_auc), lw=2.0)

            F.write('Accuracy: {0:.4f}%\n'.format(np.mean(accuray)))
            # print('auROC: {0:.6f}'.format(np.mean(auROC)))
            F.write('auROC: {0:.4f}\n'.format(mean_auc))
            # F.write('AUC: {0:.4f}\n'.format( np.mean(AUC)))
            F.write('auPR: {0:.4f}\n'.format(np.mean(avePrecision)))  # average_Precision
            F.write('F1_Score: {0:.4f}\n'.format(np.mean(F1_Score)))
            F.write('MCC: {0:.4f}\n'.format(np.mean(MCC)))

            TN, FP, FN, TP = CM.ravel()
            F.write('Recall: {0:.4f}\n'.format(np.mean(Recall)))
            F.write('Sensitivity: {0:.4f}%\n'.format(((TP) / (TP + FN)) * 100.0))
            F.write('Specificity: {0:.4f}%\n'.format(((TN) / (TN + FP)) * 100.0))
            F.write('Confusion Matrix:\n')
            F.write(str(CM) + '\n')
            F.write('_______________________________________' + '\n')

    ##########
    F.close()
    ##########

    ### auROC Curve ###
    if args.auROC == 1:
        auROCplot()
    ### boxplot algorithm comparison ###
    if args.boxPlot == 1:
        boxPlot(Results, Names)
    ### --- ###

    print('\nPlease, eyes on evaluationResults.txt')


def boxPlot(Results, Names):
    ### Algoritms Comparison ###
    # boxplot algorithm comparison
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.rcParams['savefig.dpi'] = 1200  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    bwith = 0.5  # 边框宽度设置为2
    TK = plt.gca()  # 获取边框
    TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
    TK.spines['left'].set_linewidth(bwith)  # 图框左边
    TK.spines['top'].set_linewidth(bwith)  # 图框上边
    TK.spines['right'].set_linewidth(bwith)  # 图框右边
    # plt.subplot()
    fig = plt.figure()
    # fig.suptitle('Classifier Comparison')
    ax = fig.add_subplot(111)
    ax.yaxis.grid(True)
    plt.boxplot(Results, patch_artist=True, vert=True, whis=True, showbox=True)
    ax.set_xticklabels(Names, fontsize=12)
    plt.xlabel('Classifiers', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.savefig('A_test_Accuracy_boxPlot.png', dpi=300)
    plt.show()
    ### --- ###


def auROCplot():
    ### auROC ###
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.rcParams['savefig.dpi'] = 1200  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    bwith = 0.5  # 边框宽度设置为2
    TK = plt.gca()  # 获取边框
    TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
    TK.spines['left'].set_linewidth(bwith)  # 图框左边
    TK.spines['top'].set_linewidth(bwith)  # 图框上边
    TK.spines['right'].set_linewidth(bwith)  # 图框右边
    # plt.subplot()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig('A_test_auROC.png', dpi=300)
    plt.show()
    ### --- ###


if __name__ == '__main__':
    # print('Please, enter number of cross validation:')
    import argparse

    p = argparse.ArgumentParser(description='Run Machine Learning Classifiers.')

    # p.add_argument('-cv', '--nFCV', type=int, help='Number of crossValidation', default=10)
    p.add_argument('-data', '--dataset', type=str, help='~/dataset.csv', default='optimumDataset.csv')
    p.add_argument('-test_data', '--test_dataset', type=str, help='~/dataset.csv', default='optimumDataset.csv')
    p.add_argument('-roc', '--auROC', type=int, help='Print ROC Curve', default=1, choices=[0, 1])
    p.add_argument('-box', '--boxPlot', type=int, help='Print Accuracy Box Plaot', default=1, choices=[0, 1])

    args = p.parse_args()

    runClassifiers(args)
