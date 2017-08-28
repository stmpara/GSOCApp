#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:34:52 2017

@author: TiffMin
"""

##Feature Selection Code
import numpy as np
try:
    from . import utils
except:
    import utils
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import VarianceThreshold
from copy import copy, deepcopy
from sklearn.preprocessing import label_binarize
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from inspect import signature
import pandas as pd

#[X, Y, X_train, X_test, y_train, y_test, geneIDList, DescList]  = utils.getDataAll("/Volumes/Transcend/PythonPackgeGSOC/Datasets/LuncClinical.csv", "/Volumes/Transcend/PythonPackgeGSOC/Datasets/geneAnalysis.csv")

#==============================================================================
# a = np.array([[1.0, 0.0, 1.0], [0.9, 0.1, 1.1]])
# b = deepcopy(a)
# sel = VarianceThreshold(threshold=(0.0025))
# a = sel.fit_transform(a)
# #a = np.array([[1.0, 0.1, 2.0], [0.9, 0.1, 1.1]])
# print(a)
# print(b)
#==============================================================================

##1. Variance Threshold
#==============================================================================
def VarianceThresholdFunction(X_train, X_test, thresholdn):
    X_trainvarth = np.empty_like(X_train)
    #X_testvarth = np.empty_like(X_test)
    X_trainvarth[:] = X_train
    #X_testvarth[:] = X_test
    #X_trainvarth, X_testvarth = deepcopy(X_train, X_test)
    #0.9
    
    
    sel = VarianceThreshold(threshold=thresholdn)
    X_trainvarth = sel.fit_transform(X_train)
    idxs1 = sel.get_support(indices=True)
    #X_testvarth = sel.fit_transform(X_test)
    #idxs2 = sel.get_support(indices=True)
    X_testvarth = np.empty_like(X_test[:,idxs1])
    X_testvarth[:,:] = X_test[:,idxs1] 
    return X_trainvarth, X_testvarth, idxs1, sel#, idxs2

##Code that does socre

#a, b,c = VarianceThresholdFunction(X_train, X_test, 1.0)
#yeah = VarianceThreshold(1.0).fit_transform(X_train, None)
#sig = signature(VarianceThreshold(1.0).fit_transform)
#params = sig.parameters 
#print(len(params))
#a.shape
#X_train.shape
##Out of 11095 features, 11064 selected with 0.6

#####Write a function that figures out indices

#==============================================================================
##2. Univariate Feature Selection

#==============================================================================
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#X.shape
#(150, 4)
def UnivFeatureFunction(X_train, X_test, y_train, y_test, n):
    traintest = SelectKBest(score_func=chi2, k=n)
    #testtest = SelectKBest(score_func=chi2, k=n)
    #fittrain = traintest.fit(X_train, y_train)
    #fittest = testtest.fit(X_test, y_test)
    traintest.fit(X_train, y_train)
    #testtest.fit(X_test, y_test)
    numpy.set_printoptions(precision=3)
    #print(fittrain.scores_)
    #print(fittest.scores_)
    featurestrain = traintest.transform(X_train)
    idxs1 = traintest.get_support(indices=True)
    X_testvarth = np.empty_like(X_test[:,idxs1])
    X_testvarth[:,:] = X_test[:,idxs1] 
    #featurestest = testtest.transform(X_test)
    ###traintest is really sel
    return featurestrain, X_testvarth, traintest.scores_, traintest.scores_, idxs1, traintest 

#a,b,c,d = UnivFeatureFunction(X_train, X_test, y_train, y_test, 15)

#3.PCA
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score

def compute_scoresPCA(X, n_components):
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))

    return pca_scores, fa_scores

def PCAFunc(X_train, X_test,y_train, y_test, n):
    pcatrain = PCA(n)
    #pcatest = PCA(n)
    fittrain = pcatrain.fit(X_train)
    #fittest = pcatest.fit(X_test)
    X_trainNew = fittrain.transform(X_train)
    X_testNew = fittrain.transform(X_test)
    #print("Explained Variance of train: %s") % fittrain.explained_variance_ratio_
#==============================================================================
#     pca_trainscores, pca_testscores = [], []
#     for i in range(n):
#         pca_trainscores.append(np.mean(cross_val_score(pcatrain, fittrain)))
#         pca_testscores.append(np.mean(cross_val_score(pcatest, fittest)))
#==============================================================================
    #print("Explained Variance of test: %s") % fittest.explained_variance_ratio_
    #return fittrain.components_, fittest.components_ 
    return X_trainNew, X_testNew, fittrain

#d,e= PCAFunc(X_train, X_test,y_train, y_test, 15)
# 4. Feature Importance
from sklearn.ensemble import ExtraTreesClassifier
# load data

# feature extraction
def FeatureImportance(X_train, X_test,y_train, y_test, n):  
    modeltrain = ExtraTreesClassifier(n_estimators = n)
    #modeltest = ExtraTreesClassifier(n_estimators = n)
    modeltrain.fit(X_train, y_train)
    #modeltest.fit(X_test, y_test)
    return modeltrain.fit_transform(X_train, y_train), modeltrain.fit_transform(X_test, y_test), modeltrain.feature_importances_


#a,b = FeatureImportance(X_train, X_test, y_train, y_test, 15)

##Calculate AUC given particular arguments
def calculateROC(featurefunc, args):
    pass

#num_classes = 4

##Estimate AUC with estimator 1 - E[f(pos)] E[1 - f(neg)]
##
total = {}
def estimateROCGivenClassifier(y_testProcessed, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_testProcessed.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])  
    #1 - E[f(pos)] E[1 - f(neg)]
    return 1- np.mean(fpr["micro"]) * np.mean(1-tpr["micro"])
        
###Prints out ROC given classifier with given X_train 
def estimateROCGivenClassifierWithCL(X_trainProcessed, X_testProcessed, y_trainProcessed, y_testProcessed, cl):
    
    classifier = OneVsRestClassifier(cl)
    y_score = classifier.fit(X_trainProcessed, y_trainProcessed).predict_proba(X_testProcessed)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_testProcessed.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])  
    #1 - E[f(pos)] E[1 - f(neg)]
    return 1- np.mean(fpr["micro"]) * np.mean(1-tpr["micro"])


###sel이란 VarianceThreshold(1.0) 같은 것

def estimateROCGivenClassifierWithCLAndSel(X_trainOr, X_testOr, y_trainOr, y_testOr, cl, sel):
#==============================================================================
#     sig = signature(VarianceThreshold(1.0)).fit_transform
#     params = sig.parameters 
#     len(params)
#     sel.fit_transform
#==============================================================================
    
    ################################
    X_trainvarth = np.empty_like(X_trainOr)
    X_trainvarth[:] = X_trainOr
    X_trainProcessed = sel.fit_transform(X_trainOr, y_trainOr)
    idxs1 = sel.get_support(indices=True)
    X_testProcessed = np.empty_like(X_testOr[:,idxs1])
    X_testProcessed[:,:] = X_testOr[:,idxs1] 
    ####################################
    ##Make y_trainProcessed and y_testProcessed
    #y_trainProcessed = np.empty_like(y_trainOr[:,idxs1])
    #y_testProcessed = np.empty_like(y_testOr[:,idxs1])
    y_trainProcessed = y_trainOr
    y_testProcessed = y_testOr
    
    classifier = OneVsRestClassifier(cl)
    y_score = classifier.fit(X_trainProcessed, y_trainProcessed).predict_proba(X_testProcessed)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_testProcessedNew = label_binarize(y_testProcessed, classes=[0, 1, 2, 3])
    
    #print(y_score[0])
    #print(y_testProcessed[0])
    #newYScore = []
    #for index in range(len(y_score)):
    #    newYScore.append(np.mean(y_score[index]))
    #newYScore = np.array(newYScore)
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_testProcessedNew.ravel(),y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])  
    #1 - E[f(pos)] E[1 - f(neg)]
    roc = 1- np.mean(fpr["micro"]) * np.mean(1-tpr["micro"])
    return roc, idxs1

from sklearn.ensemble import RandomForestClassifier

#cl = RandomForestClassifier(n_estimators = 50)
#sel = SelectKBest(score_func=chi2, k=10)
#r, i = estimateROCGivenClassifierWithCLAndSel(X_train, X_test, y_train, y_test, cl, sel)

### 전체 Output Function
## outputs dataframe which is 

## inputs are user inputs and data: classifier, feature selection, X_train, X_test, y_train, y_test
##Or maybe could just start from file1, file2
def totalOutputFunctionWithParams(X_trainOr, X_testOr, y_trainOr, y_testOr, cl, sel):
    roc, idxs1 = estimateROCGivenClassifierWithCLAndSel(X_trainOr, X_testOr, y_trainOr, y_testOr, cl, sel)
    
    df = pd.DataFrame.from_dict({})
    
    return df

def totalOutputFunction(file1, file2, cl, sel):
    [X, Y, X_trainOr, X_testOr, y_trainOr, y_testOr, geneIDList, DescList] = utils.getDataAll(file1, file2)
    roc, idxs1 = estimateROCGivenClassifierWithCLAndSel(X_trainOr, X_testOr, y_trainOr, y_testOr, cl, sel)
    columns = [geneIDList[a] for a in idxs1]
    desc = [DescList[a] for a in idxs1]
    FeatureImp = np.ones(len(idxs1))
    
    ##If feature importance returned as the last two? or so arguments
    ##Make Feature importance something not ones
    #if sel 
    
    dfDict = {'Feature Genes': columns,
              'Descriptions': desc,
              'Feature Importance': FeatureImp}
    
    
#==============================================================================
#     
#     
#     df2 = pd.DataFrame({ 'A' : 1.,
#    ....:                      'B' : pd.Timestamp('20130102'),
#    ....:                      'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
#    ....:                      'D' : np.array([3] * 4,dtype='int32'),
#    ....:                      'E' : pd.Categorical(["test","train","test","train"]),
#    ....:                      'F' : 'foo' })
#    ....: 
# 
# In [11]: df2
# Out[11]: 
#    A          B  C  D      E    F
# 0  1 2013-01-02  1  3   test  foo
# 1  1 2013-01-02  1  3  train  foo
# 2  1 2013-01-02  1  3   test  foo
# 3  1 2013-01-02  1  3  train  foo
#     
#==============================================================================
    
    #dfDict["roc"] = roc
    
    ###How should the dictionary look?
    
    df = pd.DataFrame.from_dict(dfDict)
    
    return df

def totalOutputWithData( X_trainOr, X_testOr, y_trainOr, y_testOr, geneIDList, DescList, cl, sel, selTicker):
    #roc, idxs1 = estimateROCGivenClassifierWithCLAndSel(X_trainOr, X_testOr, y_trainOr, y_testOr, cl, sel)
    ##DOnt use the above can just get idxs1 thru other means
    if selTicker == "pca":
        columns = [str(i)+"th direction" for i in range(X_trainOr.shape[1])]
        #sel.get_covariance(X)[
        desc = columns
        #desc = sel.components_
        #desc = sel.score_samples(X_trainOr)
        

    else:
        idxs1 = sel.get_support(indices=True)
        columns = [geneIDList[a] for a in idxs1]
        desc = [DescList[a] for a in idxs1]
    FeatureImp = np.ones(X_trainOr.shape[1])
    
    ##If feature importance returned as the last two? or so arguments
    ##Make Feature importance something not ones
    #if sel 
    
    dfDict = {'Feature Genes': columns,
              'Descriptions': desc,
              'Feature Importance': FeatureImp}
    
    
#==============================================================================
#     
#     
#     df2 = pd.DataFrame({ 'A' : 1.,
#    ....:                      'B' : pd.Timestamp('20130102'),
#    ....:                      'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
#    ....:                      'D' : np.array([3] * 4,dtype='int32'),
#    ....:                      'E' : pd.Categorical(["test","train","test","train"]),
#    ....:                      'F' : 'foo' })
#    ....: 
# 
# In [11]: df2
# Out[11]: 
#    A          B  C  D      E    F
# 0  1 2013-01-02  1  3   test  foo
# 1  1 2013-01-02  1  3  train  foo
# 2  1 2013-01-02  1  3   test  foo
# 3  1 2013-01-02  1  3  train  foo
#     
#==============================================================================
    
    #dfDict["roc"] = roc
    
    ###How should the dictionary look?
    
    df = pd.DataFrame.from_dict(dfDict)
    
    return df


#from sklearn.ensemble import RandomForestClassifier
#cl = RandomForestClassifier(n_estimators = 50)
#sel = sel = VarianceThreshold(1.0)
#dfYes = totalOutputFunction("/Volumes/Transcend/PythonPackgeGSOC/Datasets/LuncClinical.csv", "/Volumes/Transcend/PythonPackgeGSOC/Datasets/geneAnalysis.csv", cl, sel)


##Try for batches
def maximizeROC(featurefunc, args):
    #featurefunc(*args)
    #tf.metrics.auc()
    #1 - E[f(pos)] E[1 - f(neg)]
    pass

#a, b = FeatureImportance(X_train, X_test,y_train, y_test, 15)

#def selectFeaturesAndROCforSVM(featurefunc,args): 
    