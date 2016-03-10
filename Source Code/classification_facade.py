
from sklearn.cross_validation import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import svm
import numpy as np
import random
#import sys
#sys.path.append('/Users/MAHMADI/Bank/PhD/xgboost-master/wrapper/wrapper')
import xgboost as xgb
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
#import skflow
from sklearn.grid_search import GridSearchCV
from handle_io import io
#from Measures import multiclass_log_loss
from pandas import read_csv
from settings import *
from feature_selection_facade import select_features_tree
import datetime, gzip
from csv import writer
from itertools import izip
import pandas as pd


def multiclass_log_loss(y_true, y_pred, eps = 1e-15):
    predictions = np.clip(y_pred, eps, 1 - eps)
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))

    return -1.0 / rows * vsota


def classification(file_path, select = False, bagging = False, test = 'cv'):
    dataSet = read_csv(file_path, delimiter=',')
    class_lable = read_csv(TRAIN_ID_PATH, delimiter=',')
    new_idx = np.argsort(class_lable.ix[:,0])
    hash_codes = class_lable.ix[new_idx,0]
    hash_codes = hash_codes.reset_index(drop=True)
    data = dataSet.ix[:,:-1].as_matrix()
    headers = dataSet.ix[:,:-1].columns.values
    #print len(headers), headers
    class_lables = dataSet.ix[:,-1].as_matrix()
    if select == True:
        data, feature_names = select_features_tree(data, class_lables, feature_names = headers)
        #print len(feature_names), feature_names

    if test == 'cv':
        custom_cross_validation(data = data, class_labels = class_lables, hash_codes = hash_codes.as_matrix(),
                                print_mc_samples = True, bagging = bagging)
    else:
        test_dataSet = read_csv(test, delimiter=',')
        test_class_label = read_csv(TEST_ID_PATH, delimiter=',')
        test_new_idx = np.argsort(test_class_label.ix[:,0])
        test_hash_codes = test_class_label.ix[test_new_idx,0]
        test_hash_codes = test_hash_codes.reset_index(drop=True)

        if select == True:
            test_data = test_dataSet[feature_names]
            #print len(test_data.columns.values), test_data.columns.values
            diff = list(set(feature_names) - set(test_data.columns.values))
            print len(diff), diff
            test_data = test_data.as_matrix()
        else:
            test_data = test_dataSet.as_matrix()
        train_test_validation(train_data = data, train_id = class_lables, test_data = test_data,
                              test_hash= test_hash_codes.as_matrix(), bagging = bagging)


def custom_cross_validation(data, class_labels, classifier_name = 'xgb', n_folds = 5, hash_codes = [],
                            print_mc_samples = False, bagging = False):
    kfold = StratifiedKFold(class_labels, n_folds = n_folds, shuffle = True, random_state = 1000)

    class_label_ids, name_dictionary = convert_class_int(class_labels)

    predicted = []
    original = []
    log_losses = []
    accuracies = []
    f1_scores = []
    fold_counter = 0
    misclassified_counter = 1
    for train_index, test_index in kfold:
        fold_counter += 1
        train_data = data[train_index, :]
        train_id = class_labels[train_index]
        #train_id_num = class_label_ids[train_index]
        num_classes = len(np.unique(train_id))

        test_data = data[test_index, :]
        test_actual_labels = class_labels[test_index]
        test_id_num = class_label_ids[test_index]
        #class_dict_temp = dict(zip(test_id_num, test_actual_labels))
        test_MD5 = hash_codes[test_index]
        #test_data_mis = data[test_index]

        if bagging is True:
            bags = 10
            test_pred_prob = np.zeros((len(test_index), num_classes))
            #new_train_index = []
            for bg in range(bags):
                seed = bg + 1
                new_train_index = random.sample(train_index, int(len(train_index) * 1.))

                for i in range(int(len(train_index) * 1.)):
                    new_train_index.append(random.choice(train_index))

                train_data = data[new_train_index, :]
                train_id = class_labels[new_train_index]

                if classifier_name == 'rf':
                    pred_class, pred_prob = randomforest_cla(train_data, train_id, test_data, seed= seed)
                elif classifier_name == 'xgb':
                    pred_class, pred_prob = xgboost_cla(train_data, train_id, test_data, seed= seed)
                elif classifier_name == 'et':
                    pred_class, pred_prob = extratree_cla(train_data, train_id, test_data, seed= seed)
                else:
                    print('Name of classifier should be : rf or xgb or et')
                    return

                print "Bag ", bg+1, " Finished!!"

                test_pred_prob += pred_prob.reshape((len(test_index), num_classes))

            pred_prob = test_pred_prob / bags

            pred_class = np.zeros(len(test_index))
            for i in range(len(test_index)):
                pred_class[i] = list(test_pred_prob[i]).index(max(test_pred_prob[i]))

            pred_class = [int(x+1) for x in pred_class]

            #print len(pred_class), pred_class

        else:

            if classifier_name == 'rf':
                pred_class, pred_prob = randomforest_cla(train_data, train_id, test_data)
            #elif classifier_name == 'svm':
            #    pred_class, pred_prob = svm_cla(train_data, train_id, test_data)
            elif classifier_name == 'xgb':
                pred_class, pred_prob = xgboost_cla(train_data, train_id, test_data)
            elif classifier_name == 'et':
                pred_class, pred_prob = extratree_cla(train_data, train_id, test_data)
            #elif classifier_name == 'dnn':
            #    pred_class1, pred_prob = tensor_dnn_cla(train_data, train_id_num, test_data)
            #    pred_class = [class_dict_temp[pred_class1[i]] for i in range(len(pred_class1))]
            else:
                print('Name of classifier should be : rf or xgb or et')
                return

            #print len(pred_class), pred_class

        predicted.extend(pred_class)
        original.extend(class_labels[test_index])

        #if pred_prob != None:
        log_losses.append(multiclass_log_loss(test_id_num, pred_prob))
        #else:
        #    log_losses.append(np.inf)
        accuracies.append(accuracy_score(test_actual_labels, pred_class))
        f1_scores.append(f1_score(test_actual_labels, pred_class, average='weighted'))
        print('-----------------------------------')
        print('FOLD '+ fold_counter)
        print accuracies[-1], f1_scores[-1], log_losses[-1]

        if(print_mc_samples == True):
            for i in range(len(pred_class)):
                if pred_class[i] != test_actual_labels[i]:
                    print misclassified_counter,'==>', test_MD5[i], 'actual='+str(test_actual_labels[i]), \
                           'predicted='+str(pred_class[i]), 'actual probability='+ str(pred_prob[i, test_id_num[i]]), \
                            'predicted probability='+ str(pred_prob[i, name_dictionary[pred_class[i]]])
                    misclassified_counter += 1

    predicted = np.array(predicted)
    original = np.array(original)
    print '==========================================='
    print('In total {} misclassified samples'.format(misclassified_counter-1))
    print('Accuracy mean : ' + str(np.mean(accuracies)))
    print('F1_score mean : ' + str(np.mean(f1_scores)))
    print('log loss mean : ' + str(np.mean(log_losses)))

    #return predicted, original
    draw_cm(original, predicted, classifier_name)


def train_test_validation(train_data, train_id, test_data, features_name = [], classifier_name = 'xgb',
                          test_id = [], test_hash = [], bagging = True):

    train_id_num = convert_class_int(train_id)
    train_id_dict = dict(zip(train_id, train_id_num))
    #test_id_num = []
    #for id in test_id:
    #    if id in train_id_dict:
    #        test_id_num.append(train_id_dict[id])
    #    else:
    #        test_id_num.append(-1)
    #test_id_num = np.array(test_id_num)

    #print train_data
    #print '+++++++++++++++++++++'
    #print test_data
    #print '+++++++++++++++++++++'
    #print test_hash

    predicted = []
    original = []
    log_losses = []
    accuracies = []
    f1_scores = []
    num_classes = len(np.unique(train_id))

    train_indices = [i for i in range(len(train_data))]

    if bagging is True:
        bags = 10
        test_pred_prob = np.zeros((len(test_data), num_classes))
        for bg in range(bags):
            seed = bg + 1
            new_train_index = random.sample(train_indices, int(len(train_indices) * 1.))

            for i in range(int(len(train_indices) * 1.)):
                new_train_index.append(random.choice(train_indices))

            train_data = train_data[new_train_index, :]
            train_id = train_id[new_train_index]

            if classifier_name == 'rf':
                pred_class, pred_prob = randomforest_cla(train_data, train_id, test_data, seed= seed)
            elif classifier_name == 'xgb':
                pred_class, pred_prob = xgboost_cla(train_data, train_id, test_data, seed= seed)
            elif classifier_name == 'et':
                pred_class, pred_prob = extratree_cla(train_data, train_id, test_data, seed= seed)
            else:
                print('Name of classifier should be : rf or xgb or et')
                return

            print "Bag ", bg+1, " Finsihed!!"
            test_pred_prob += pred_prob.reshape((len(test_data), num_classes))

        pred_prob = test_pred_prob / bags

        pred_class = np.zeros(len(test_data))
        for i in range(len(test_data)):
            pred_class[i] = list(test_pred_prob[i]).index(max(test_pred_prob[i]))

        pred_class = [int(x+1) for x in pred_class]

        #print len(pred_class), pred_class

    else:

        if classifier_name == 'rf':
            pred_class, pred_prob = randomforest_cla(train_data, train_id, test_data)
        #elif classifier_name == 'svm':
        #    pred_class, pred_prob = svm_cla(train_data, train_id, test_data)
        elif classifier_name == 'xgb':
            pred_class, pred_prob = xgboost_cla(train_data, train_id, test_data)
        elif classifier_name == 'et':
            pred_class, pred_prob = extratree_cla(train_data, train_id, test_data)
        #elif classifier_name == 'dnn':
        #    pred_class1, pred_prob = tensor_dnn_cla(train_data, train_id_num, test_data)
        #    pred_class = [class_dict_temp[pred_class1[i]] for i in range(len(pred_class1))]
        else:
            print('Name of classifier should be : rf or xgb or et')
            return

    submission_file = SUB_PATH + 'submission'+ str(datetime.datetime.now()) + '.gz'
    with gzip.open(submission_file, 'w') as f:
        fw = writer(f)
        # Header preparation
        header = ['Id'] + ['Prediction'+str(i) for i in range(1,10)]
        fw.writerow(header)
        for t, (Id, pred) in enumerate(izip(test_hash, pred_prob.tolist())):
            fw.writerow([Id]+pred)
            if(t+1)%1000==0:
                print(t+1 + ' predictions written!!')


def draw_cm(original, predicted, classifier_name):
    cm = confusion_matrix(original, predicted)
    names = np.unique(original)
    plot_confusion_matrix_fancy(cm, title='Confusion matrix on MicMalChal '+classifier_name, names=names)


def plot_confusion_matrix_fancy(conf_arr, title='Confusion matrix', names=[]):
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure(figsize(10, 10))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Blues,
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.title(title)
    cb = fig.colorbar(res)
    plt.xticks(range(width), names, rotation='vertical')
    plt.yticks(range(height), names)
    plt.savefig(DATASET_PATH+title+'.png', format='png', dpi=200)


def convert_class_int(class_labels):
    unique_names = np.unique(class_labels)
    unique_id = range(len(unique_names))
    name_dictionary = dict(zip(unique_names, unique_id))
    return_list = []
    for i in range(len(class_labels)):
        return_list.append(name_dictionary[class_labels[i]])
    return np.array(return_list), name_dictionary


def xgboost_cla(train_data, train_id, test_data, seed = None):
    #clf = xgb.XGBClassifier(learning_rate=0.5, n_estimators=50, max_depth=5, nthread=6, seed=seed)

    train_id = [int(x-1) for x in train_id]

    param = {}
    param['booster'] = 'gbtree'
    param['objective'] = 'multi:softprob'
    param['num_class'] = 9
    param['eval_metric'] = 'logloss'
    param['scale_pos_weight'] = 1.0
    param['bst:eta'] = 0.5
    param['bst:max_depth'] = 5
    param['bst:colsample_bytree'] = 0.5
    param['silent'] = 1
    param['nthread'] = 8
    param['seed'] = seed

    num_round = 50
    watchlist = []
    plst = list(param.items())

    Xdatatrain = xgb.DMatrix(data = train_data, label = train_id)
    Xdatatest = xgb.DMatrix(data = test_data)

    clf = xgb.train(plst, Xdatatrain, num_round, watchlist)

    pred_prob = clf.predict(Xdatatest)

    pred_class = np.zeros(len(test_data))
    for i in range(len(test_data)):
        pred_class[i] = list(pred_prob[i]).index(max(pred_prob[i]))

    pred_class = [int(x+1) for x in pred_class]

    return pred_class, pred_prob


def randomforest_cla(train_data, train_id, test_data, seed = None):
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=4, random_state= seed)
    clf.fit(train_data, train_id)
    pred_class = clf.predict(test_data)
    pred_prob = clf.predict_proba(test_data)
    return pred_class, pred_prob


def extratree_cla(train_data, train_id, test_data, seed = None):
    clf = ExtraTreesClassifier(n_estimators=1000, n_jobs=4, random_state= seed)#, max_features="log2")
    param_grid = {
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
    }
    clf.fit(train_data, train_id)
    pred_class = clf.predict(test_data)
    pred_prob = clf.predict_proba(test_data)
    return pred_class, pred_prob

