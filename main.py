#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File    ：master_thesis -> main
@IDE    ：PyCharm
@Author  : Xue-Qiang Fan
@Date   ：2022/3/25 16:14
=================================================='''
import warnings

warnings.filterwarnings('ignore')
import argparse
from I_DNAN6mAplus import save
from I_DNAN6mAplus import generate_features
from I_DNAN6mAplus import select_important_features
# from I_DNAN6mAplus.runnClassifiers import runClassifiers
# from I_DNAN6mAplus.runnClassifiers_averaging_voting import runClassifiers
# from I_DNAN6mAplus_D.runnClassifiers_averaging_voting_test import runClassifiers
# from I_DNAN6mAplus_D.runnClassifiers_2_averaging_voting_test import runClassifiers
def main(args):
    X, Y = generate_features.read_seq_label(args.train_data)
    print('\nDatasets fetching done.')
    print('Features extraction begins. Be patient! The machine will take some time.')
    T = generate_features.gF(args, X, Y)
    X_train = T[:, :-1]
    Y_train = T[:, -1]
    print('Features extraction ends.')
    print('[Total extracted feature: {}]\n'.format(X_train.shape[1]))

    print('Select Important Feature begins. Be patient! The machine will take some time.')
    select_important_features.selectKImportanceFeatures(X_train, Y_train)
    print('Features Selection ends.')
    X_train = select_important_features.accordBestIndexExtractFeature(args.best_index, X_train)
    print('[Total Selected feature: {}]\n'.format(X_train.shape))

    X, Y = generate_features.read_seq_label(args.test_data)
    print('Features extraction begins. Be patient! The machine will take some time.')
    T = generate_features.gF(args, X, Y)
    X_test = T[:, :-1]
    Y_test = T[:, -1]
    print('[Total extracted feature: {}]\n'.format(X_test.shape[1]))

    X_test = select_important_features.accordBestIndexExtractFeature(args.best_index, X_test)
    print('[Total Selected feature: {}]\n'.format(X_test.shape))

    runClassifiers(args, X_train, Y_train, X_test, Y_test)


if __name__ == '__main__':
    ######################
    # Adding Arguments
    #####################
    p = argparse.ArgumentParser(description=''' step 1. Features Generation From DNA Sequences
                                                step 2. Selection Important Feature 
                                                step 3. Run Machine Learning and Deep Learning Classifiers ''')

    # p.add_argument('-cv', '--nFCV', type=int, help='Number of crossValidation', default=10)
    p.add_argument('-train_data', '--train_data', type=str, help='~/dataset.csv')
    p.add_argument('-test_data', '--test_data', type=str, help='~/dataset.csv')
    p.add_argument('-roc', '--auROC', type=int, help='Print ROC Curve', default=1, choices=[0, 1])
    p.add_argument('-box', '--boxPlot', type=int, help='Print Accuracy Box Plaot', default=1, choices=[0, 1])
    p.add_argument('-seq', '--sequenceType', type=str, help='DNA', default='DNA')
    p.add_argument('-best_index', '--best_index', type=str, help='~/selectIndex.csv', default="selectedIndex.csv")
    p.add_argument('-kgap', '--kGap', type=int, help='(l,k,p)-mers', default=5)
    p.add_argument('-ktuple', '--kTuple', type=int, help='k=1 then (X), k=2 then (XX), k=3 then (XXX),', default=3)
    p.add_argument('-full', '--fullDataset', type=int, help='saved full dataset', default=0, choices=[0, 1])
    p.add_argument('-test', '--testDataset', type=int, help='saved test dataset', default=0, choices=[0, 1])
    p.add_argument('-optimum', '--optimumDataset', type=int, help='saved optimum dataset', default=1, choices=[0, 1])

    p.add_argument('-one_hot_encoding', '--one_hot_encoding', type=int, help='Generate feature: one hot encoding',
                   default=1, choices=[0, 1])
    p.add_argument('-pseudo', '--pseudoKNC', type=int, help='Generate feature: X, XX, XXX, XXX', default=1,
                   choices=[0, 1])
    p.add_argument('-zcurve', '--zCurve', type=int, help='x_, y_, z_', default=1, choices=[0, 1])
    p.add_argument('-gc', '--gcContent', type=int, help='GC/ACGT', default=1, choices=[0, 1])
    p.add_argument('-f11', '--monoMono', type=int, help='Generate feature: X_X', default=1, choices=[0, 1])
    p.add_argument('-f12', '--monoDi', type=int, help='Generate feature: X_XX', default=1, choices=[0, 1])
    p.add_argument('-f21', '--diMono', type=int, help='Generate feature: XX_X', default=1, choices=[0, 1])

    p.add_argument('-skew', '--cumulativeSkew', type=int, help='GC, AT', default=0, choices=[0, 1])
    p.add_argument('-atgc', '--atgcRatio', type=int, help='atgcRatio', default=0, choices=[0, 1])
    p.add_argument('-f13', '--monoTri', type=int, help='Generate feature: X_XXX', default=0, choices=[0, 1])
    p.add_argument('-f22', '--diDi', type=int, help='Generate feature: XX_XX', default=0, choices=[0, 1])
    p.add_argument('-f23', '--diTri', type=int, help='Generate feature: XX_XXX', default=0, choices=[0, 1])
    p.add_argument('-f31', '--triMono', type=int, help='Generate feature: XXX_X', default=0, choices=[0, 1])
    p.add_argument('-f32', '--triDi', type=int, help='Generate feature: XXX_XX', default=0, choices=[0, 1])

    args = p.parse_args()

    main(args)
