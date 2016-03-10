

import os
from pandas.io.parsers import read_csv
from Measures import multiclass_log_loss
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
from dataSetInfo import dataSetInformaion
from settings import *
from handle_io import io
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier



def feature_fusion():
    # Name of train file
    trainName = '/SortedTrain.csv'
    # Path of a folder contains all single feature categories
    savePath = COMBINED_PATH_CSV
    path = SAVED_PATH_CSV + 'train/'
    os.chdir(path)
    # Read the name of each folder
    featureCategoryFolders = io.get_files_in_directory(path ,file_extension='csv')
    #featureCategoryFolders = featureCategoryFolders[1:]
    print 'Folders: ', featureCategoryFolders
    # Load train files
    featureCategoriesLen = len(featureCategoryFolders)
    class_lable = read_csv(TRAIN_ID_PATH, delimiter=',')
    new_idx = np.argsort(class_lable.ix[:,0])
    class_lable = class_lable.ix[new_idx,1]
    #print(class_lable[700:800])
    class_lable = class_lable.reset_index(drop=True)
    singleTrains = list()
    for k in range(featureCategoriesLen):
        dataSet = read_csv(featureCategoryFolders[k], delimiter=',')
        data = dataSet#.ix[:, :]
        #data.astype(np.float)
        #print(class_lable[0:data.shape[0]-1])
        singleTrain = dataSetInformaion(featureCategoryFolders[k][featureCategoryFolders[k].rfind('/')+1:-4],
                                        data, class_lable[0:data.shape[0]])
        singleTrains.append(singleTrain)

    remainingFeatureCategoriesIndices = np.ones(len(featureCategoryFolders))
    finalDataSets = list()
    # Main loop of combination
    for A in range(featureCategoriesLen):

        minimumLogLoss = 100
        minimumDataSet = None
        minimumIndex = -1
        processingDataSet = None
        # Main loop of single evaluation
        for k in range(featureCategoriesLen):
            nameOfDataSet = ''
            if remainingFeatureCategoriesIndices[k] == 0:
                continue
            if len(finalDataSets) != 0:
                # Join datasets
                dataSet1 = finalDataSets[len(finalDataSets) - 1].data
                # print dataSet1
                dataSet2 = singleTrains[k].data
                # print dataSet2
                result = pd.concat([dataSet1, dataSet2], axis=1, join='inner')

                processingDataSet = dataSetInformaion(
                    finalDataSets[len(finalDataSets) - 1].dataSetName + '+' + singleTrains[k].dataSetName, result,
                    singleTrains[k].classLabel)
                # print result
                # classLabel = result.ix[:, -1]
                # data = result.ix[:, :-1]
                # nameOfDataSet =
            else:
                processingDataSet = singleTrains[k]
                # classLabel = singleTrains[k].data
                # data = singleTrains[k].classLabel
                # nameOfDataSet = featureCategoryFolders[k]
            accuracies = []
            logLosses = []

            # print 'DataSet', str(k), '=========================', 'cross validation result'
            # # For each fold in cross validation
            # rng = np.random.RandomState(31337)
            # kF = KFold(classLabel.shape[0], n_folds=2, shuffle=True, random_state=rng)
            # for trainIndex, testIndex in kF:
            #     trainKF = data.ix[trainIndex,:]
            #     trainID = classLabel.ix[trainIndex]
            #     xgbModel = xgb.XGBClassifier().fit(trainKF,trainID)
            #     actualLabels = classLabel.ix[testIndex]
            #
            #     predictProbability = xgbModel.predict_proba(data.ix[testIndex,:])
            #     logLoss = multiclass_log_loss(actualLabels,predictProbability)
            #     logLosses.append(logLoss)
            #
            #     predictedLabels = xgbModel.predict(data.ix[testIndex,:])
            #     #print(confusion_matrix(actualLabels, predictedLabels))
            #     acc = accuracy_score(actualLabels, predictedLabels)
            #     accuracies.append(acc)
            #
            # accuraciesMean = np.mean(accuracies)
            # loglossesMean = np.mean(logLosses)
            # print accuraciesMean
            # print loglossesMean
            #
            # FeatureCategoriesAccuracy.append(accuraciesMean)
            # FeatureCategoriesLogLoss.append(loglossesMean)

            print 'DataSet', str(processingDataSet.dataSetName), '=========================', 'train result'
            # For each fold in cross validation

            xgbModel = xgb.XGBClassifier().fit(processingDataSet.data, processingDataSet.classLabel)
            predictProbability = xgbModel.predict_proba(processingDataSet.data)
            logLoss = multiclass_log_loss(processingDataSet.classLabel, predictProbability)
            #logLoss = float("%.3f"% logLoss)
            predictedLabels = xgbModel.predict(processingDataSet.data)
            acc = accuracy_score(processingDataSet.classLabel, predictedLabels)
            print acc
            print logLoss
            print processingDataSet.data.shape

            if logLoss < minimumLogLoss:
                minimumDataSet = processingDataSet
                minimumIndex = k
                minimumLogLoss = logLoss
                # featureCategoriesAccuracy.append(acc)
                # featureCategoriesLogLoss.append(logLoss)

        print 'Final Round ', A, '========================='
        # Minimum_Index = np.where(featureCategoriesLogLoss == np.min(featureCategoriesLogLoss))
        # print featureCategoriesLogLoss
        # finalDataSets.append(singleTrains[Minimum_Index[0]])
        finalDataSets.append(minimumDataSet)
        remainingFeatureCategoriesIndices[minimumIndex] = 0
    finalSets = [set.dataSetName for set in finalDataSets]
    print ','.join(finalSets)
    # Save the combined datasets
    os.chdir(savePath)
    for ds in finalDataSets:
        jointFile = pd.concat([ds.data, ds.classLabel], axis=1, join='inner')
        if not os.path.exists(savePath + ds.dataSetName):
            os.makedirs(savePath + ds.dataSetName)
        jointFile.to_csv(COMBINED_PATH_CSV + ds.dataSetName + '/NewTrain.csv', sep=',', index=False)
    try:
        del singleTrains, singleTrain, processingDataSet, jointFile, finalSets, \
            remainingFeatureCategoriesIndices, featureCategoryFolders, dataSet, dataSet1, dataSet2, data, class_lable
    except:
        pass
    print 'All combinations saved!!!'
    print 'Run cross-validation ...'
    featureCombinationsFinalAccuracy = 0
    featureCombinationsFinalDataSet = None
    featureCombinationsLogLossMin = 100
    for ds in finalDataSets:
        print 'DataSet', ds.dataSetName, '=========================', 'cross validation result'
        # For each fold in cross validation
        rng = np.random.RandomState(31337)
        kF = KFold(ds.classLabel.shape[0], n_folds=5, shuffle=True, random_state=rng)
        for trainIndex, testIndex in kF:
            trainKF = ds.data.ix[trainIndex, :]
            trainID = ds.classLabel.ix[trainIndex]
            xgbModel = xgb.XGBClassifier().fit(trainKF, trainID)
            actualLabels = ds.classLabel.ix[testIndex]

            predictProbability = xgbModel.predict_proba(ds.data.ix[testIndex, :])
            logLoss = multiclass_log_loss(actualLabels, predictProbability)
            logLosses.append(logLoss)

            predictedLabels = xgbModel.predict(ds.data.ix[testIndex, :])
            # print(confusion_matrix(actualLabels, predictedLabels))
            acc = accuracy_score(actualLabels, predictedLabels)
            accuracies.append(acc)

        accuraciesMean = np.mean(accuracies)
        loglossesMean = np.mean(logLosses)
        print accuraciesMean
        print loglossesMean

        if loglossesMean < featureCombinationsLogLossMin:
            featureCombinationsFinalDataSet = ds
            featureCombinationsFinalAccuracy = accuraciesMean
            featureCombinationsLogLossMin = loglossesMean
    print 'Final Result ---------------------------------'
    print featureCombinationsFinalDataSet.dataSetName, featureCombinationsFinalAccuracy, featureCombinationsLogLossMin


def select_features_tree(X, y, feature_names = []):
    print X.shape
    #forest = RandomForestClassifier(n_estimators=1000, n_jobs=4)
    forest = ExtraTreesClassifier(n_estimators=1000, n_jobs=8)
    fo = forest.fit(X, y)
    sorted_feature_names = plot_feature_importance(fo, X, feature_names)
    model = SelectFromModel(fo, prefit=True, )
    X_new = model.transform(X)
    print X_new.shape
    return X_new, sorted_feature_names[0:X_new.shape[1]]


def plot_feature_importance(model_name, train, feature_names):
    args = np.argsort(-model_name.feature_importances_)
    rankings = ''
    #print feature_names
    sorted_feature_names = []
    for i in range(len(args)):
        sorted_feature_names.append(feature_names[args[i]])
        #rankings += feature_names[args[i]] + '->' + str(model_name.feature_importances_[args[i]]) + '\n'

    #print rankings
    #figsize(12, 6)
    #plt.xlabel('Feature names')
    #plt.ylabel('Importance Scores')
    #plt.title('Feature importance by Random Forest')
    ###############
    #max_features = 40
    #heights = model_name.feature_importances_[args]
    #plt.figure(1)
    #h = plt.bar(xrange(len(sorted_feature_names[0:max_features])), heights[0:max_features], label=sorted_feature_names[0:max_features])
    #plt.subplots_adjust(bottom=0.6)
    #xticks_pos = [0.65 * patch.get_width() + patch.get_xy()[0] for patch in h]
    #plt.xticks(xticks_pos, sorted_feature_names[0:max_features], ha='right', rotation=45)
    #make_bar_chart(sorted_feature_names[0:max_features], heights[0:max_features])
    ################
    #plt.savefig('/Users/MAHMADI/Dropbox/phd/SP16/RFranking-' + mode + '.png', format='png', dpi=200)
    return sorted_feature_names