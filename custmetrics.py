#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:52:44 2020

@author: user

"""
import numpy as np
from math import sqrt
from sklearn.metrics import confusion_matrix, roc_curve, auc, \
                            log_loss, brier_score_loss, roc_auc_score, \
                            mean_squared_error
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

def cap_acc_rate(probs, y_test):
    '''
    Parameters
    ----------
    probs : Numpy Array of float64
        Probabilities that predicted class is 1..
    y_test : Numpy Array of int64
        Testing sample of dependent value.

    Returns
    -------
    acc_rate: float64
        Accuracy Rate based on CAP..

    '''
    total = len(y_test)
    cnt_1 = sum(y_test)
    
    x_axis = np.arange(0,total+1)
    y_axis = np.append([0], np.cumsum([b for _,b in sorted(zip(probs,y_test),reverse=True)]))
    
    #area under random curve
    rand_curve = auc([0, total], [0, cnt_1])
    
    #area under predicted curve
    pred_curve = auc(x_axis,y_axis)
    
    #area under perfect curve
    perf_curve = auc([0,cnt_1,total],[0,cnt_1,cnt_1])
    
    acc_rate = (pred_curve - rand_curve)/(perf_curve - rand_curve)
    
    return acc_rate

def draw_cap(probs, y_test, text):
    '''
    Draws Cumulative Accuracy Profile Curve.

    Parameters
    ----------
    probs : Numpy Array of float64
        Probabilities that predicted class is 1.
    y_test : Numpy Array of int64
        Testing sample of dependent value.
    text : String
        Used for labeling the curve.

    Returns
    -------
    None. Draws CAP.

    '''
    total = len(y_test)
    cnt_1 = sum(y_test)
    
    x_axis = np.arange(0,total+1)
    y_axis = np.append([0], np.cumsum([b for _,b in sorted(zip(probs,y_test),reverse=True)]))
        
    label_rand = "Random Model"
    label_pred = "Predicted Model"
    label_perf = "Pefect Model"
    
    plt.figure(figsize=(7,5))
    #random curve
    plt.plot([0, total], [0, cnt_1], c = 'r', linestyle = '--', label = label_rand)
    #predicted curve
    plt.plot(x_axis, y_axis, c = 'b',label = label_pred)
    #perfect curve
    plt.plot([0,cnt_1,total],[0,cnt_1,cnt_1], c = 'g', label = label_perf)
    
    plt.xlabel('Total number of observations', fontsize = 12)
    plt.ylabel('Number of observations with class 1', fontsize = 12)
    plt.title(text + ' Cumulative Accuracy Profile', fontsize = 12)
    plt.legend(loc = 'lower right', fontsize = 12)
    
    plt.show()
    
def cm_acc_rate(y_pred, y_test):
    '''

    Parameters
    ----------
    y_pred : Numpy Array of int64
        Values predicted by classifier based on \
        testing sample of independenet values.
    y_test : Numpy Array of int64
        Testing sample of dependent value.

    Returns
    -------
    acc_rate : float64
        Accuracy Rate based on Confusion Matrix.

    '''
    cm = confusion_matrix(y_test,y_pred)
    acc_rate = (cm[0][0]+cm[1][1])/len(y_pred)
    return acc_rate

def draw_roc(probs, y_test, text):
    '''
    Draws Receiver Operating Characteristic Curve. 
    The True Positive Rate (TPR) is plotted against False Positive Rate (FPR) 
    for the probabilities of the classifier predictions. 
    More the area under the curve, the better is the model.

    Parameters
    ----------
    probs : Numpy Array of float64
        Probabilities that predicted class is 1.
    y_test : Numpy Array of int64
        Testing sample of dependent value.
    text : String
        Used for labeling the curve.

    Returns
    -------
    None. Draws ROC Curve.

    '''
    fpr, tpr, thresholds = roc_curve(y_test,probs)
    label = text + ' AUC:' + ' {0:.2f}'.format(roc_auc_score(y_test,probs))
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, c = 'g', label = label, linewidth = 4)
    plt.xlabel('False Positive Rate', fontsize = 12)
    plt.ylabel('True Positive Rate', fontsize = 12)
    plt.title('Receiver Operating Characteristic', fontsize = 12)
    plt.legend(loc = 'lower right', fontsize = 12)
    plt.show()

def draw_errors_knn(X_train, y_train, X_test, y_test, metric = 'minkowski', p = 2):
    '''
    
    Parameters
    ----------
    X_train : Numpy Array of float64
        Training sample of independent values.
    y_train : Numpy Array of int64
        Training sample of dependent value.
    X_test : Numpy Array of float64
        Testing sample of independent values.
    y_test : Numpy Array of int64
        Testing sample of dependent value.
    metric : String or Callable, optional
        The distance metric to use for the tree. The default is 'minkowski'.
    p : Integer, optional
        Power parameter for the Minkowski metric. The default is 2 for \
        Euclidean distance.

    Returns
    -------
    None. Draws Number of Neighbours for KNN against Mean Squared Rate.

    '''
    errors = []
    cluster_num = range(1,10)
    for k in cluster_num:
        knn = KNeighborsClassifier(n_neighbors = k, metric = metric, p = p)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        errors.append(sqrt(mean_squared_error(y_test,y_pred)))

    plt.figure(figsize=(7,5))
    plt.plot(cluster_num, errors)
    plt.xlabel('Number of Neighbours')
    plt.ylabel('Mean Squared Rate')
    plt.show()
        
def calc_acc(probs, X_test, y_pred, y_test, text = 'NaN'):
    '''
    

    Parameters
    ----------
    probs : Numpy Array of float64
        Probabilities that predicted class is 1.
    X_test : Numpy Array of float64
        Testing sample of independent values.
    y_pred :  Numpy Array of int64
        Values predicted by classifier based on \
        testing sample of independenet values.
    y_test : Numpy Array of int64
        Testing sample of dependent value.
    text : String, optional
        Name of the Classifier. The default is 'NaN'.

    Returns
    -------
    scores : List
        All accuracy scores for given Classifier.

    '''
    scores = []
    
    acc_rate = cm_acc_rate(y_pred, y_test)
    cap_score = cap_acc_rate(probs, y_test)
    roc_score = roc_auc_score(y_test,probs)
    brier_score = brier_score_loss(y_test, probs)
    log_loss_score = log_loss(y_test, probs)
    
    scores.append((text
                  ,round(acc_rate,3)
                  ,round(cap_score,3)
                  ,round(roc_score,3)
                  ,round(brier_score,3)
                  ,round(log_loss_score,3)))
    
    return scores

def draw_all(probs, y_test, text):
    '''

    Parameters
    ----------
    probs : Numpy Array of float64
        Probabilities that predicted class is 1.
    y_test : Numpy Array of int64
        Testing sample of dependent value.
    text : String
        Used for labeling the curve.

    Returns
    -------
    None. Draws CAP and ROC Curves.

    '''
    draw_cap(probs, y_test, text)
    draw_roc(probs, y_test, text)

def RFECV_selector(classifier, X_col, X_train, y_train, X_test):
    '''

    Parameters
    ----------
    classifier : Instance of a Class
        Sklearn Classifier.
    X_col : Pandas Index
        Column names.
    X_train : Numpy Array of float64
        Training sample of independent values.
    y_train : Numpy Array of int64
        Training sample of dependent value.
    X_test : Numpy Array of float64
        Testing sample of independent values.

    Returns
    -------
    None. Prints optimal number of features.

    '''
    rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(3),scoring='accuracy')
    rfecv.fit(X_train, y_train)
    print("Optimal number of features : %d" % rfecv.n_features_)
    num_col = [X_col.get_loc(name) for name in X_col[rfecv.support_]]
    X_train = X_train[:,num_col]
    X_test = X_test[:,num_col]
    

def rf_scores(X_train, y_train, X_test, y_test, start=1, stop=60, step=9, \
                criterion = 'entropy', random_state = None):
    '''
    

    Parameters
    ----------
    X_train : Numpy Array of float64
        Training sample of independent values.
    y_train : Numpy Array of int64
        Training sample of dependent value.
    X_test : Numpy Array of float64
        Testing sample of independent values.
    y_test : Numpy Array of int64
        Testing sample of dependent value.
    start : Integer, optional
        Controls the starting point of interation, which runs through
        number of estimators for Random Forest Classifier. The default is 1.
    stop : Integer, optional
        Controls the stopping point of interation. The default is 60.
    step : Integer, optional
        Controls the step of interation. The default is 9.
    criterion : String, optional
        The function to measure the quality of a split. 
        Can either be 'gini' or 'entropy'. The default is 'entropy'.
    random_state : Integer, optional
        Controls the randomness. The default is None. 

    Returns
    -------
    scores_out : List of float64
        Accuracy scores for each number of estimators. Used to draw a plot.

    '''
    scores_out= []
    for i in range(start, stop, step):
        classifierRFC = RandomForestClassifier(n_estimators = i, \
                                               criterion = 'entropy', random_state = 0)
        classifierRFC.fit(X_train,y_train)
        y_pred = classifierRFC.predict(X_test)
        probs = classifierRFC.predict_proba(X_test)[:,1]
        
        acc_rate = cm_acc_rate(y_pred, y_test)
        cap = cap_acc_rate(probs, y_test)    
        roc= roc_auc_score(y_test,probs)
        
        brier_score = brier_score_loss(y_test, probs)
        log_score= log_loss(y_test, probs)
            
        scores_out.append((i,brier_score,log_score,acc_rate,cap,roc))
        
    return scores_out
    
def rf_draw_scores(scores, x_start=0, x_stop = 40):
    '''
    

    Parameters
    ----------
    scores : List of float64
        Accuracy scores for each number of estimators
        for Random Forest Classifier.
    x_start : Integer, optional
        Corresponds to starting point of iteration used
        in calculating the scores. The default is 0.
    x_stop : Integer, optional
        Corresponds to stopping point of iteration used
        in calculating the scores. The default is 40.

    Returns
    -------
    None. Draws a plot.

    '''
    f, ax = plt.subplots(3,1, figsize=(5.5,7), sharey = False, sharex = True)
    num = [i[0] for i in scores]
    brier_score = [i[1] for i in scores]
    log_score = [i[2] for i in scores]
    acc_rate = [i[3] for i in scores]
    cap = [i[4] for i in scores]
    roc = [i[5] for i in scores]
    
    label1 = "brier"
    label2 = "log"
    label3 = "acc_rate"
    label4 = "cap"
    label5 = "roc"
    
    ax[0].plot(num, brier_score, c = 'r', label = label1)
    ax[1].plot(num, log_score, c = 'black', label = label2)
    ax[2].plot(num, acc_rate, c = 'b', label = label3)
    ax[2].plot(num, cap, c = 'g', label = label4)
    ax[2].plot(num, roc, c = 'y', label = label5)
    ax[2].set(xlim = (x_start,x_stop))
    ax[0].grid(axis ='both')
    ax[1].grid(axis ='both')
    ax[2].grid(axis ='both')
    f.legend(bbox_to_anchor = (1.1,0.5), frameon = False)
    plt.show()
