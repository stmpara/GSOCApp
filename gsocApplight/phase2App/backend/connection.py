#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:00:03 2017

@author: TiffMin
"""
import numpy as np
try:
    from . import utils, roc, feature_selection
except:
    import utils, roc, feature_selection
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier

#X, Y, X_train, X_test, y_train, y_test, geneIDList, DescList  = utils.getDataAll("/Volumes/Transcend/PythonPackgeGSOC/Datasets/LuncClinical.csv", "/Volumes/Transcend/PythonPackgeGSOC/Datasets/geneAnalysis.csv")
#cl = RandomForestClassifier(n_estimators = 50)
#==============================================================================
# 
# cl = RandomForestClassifier(n_estimators = 50)
# 
# a, b,X_trains, X_tests, sel, classifier = roc.findBestVarVarThreshold(X_train, X_test, y_train, y_test, cl)
# 
# #Could plot with b
# sel = VarianceThreshold(a)
# 
# df = feature_selection.totalOutputWithData(X_train, X_test, y_train, y_test, geneIDList, DescList, cl, sel)
# 
# print(df)
#==============================================================================

def intoPreFromStart(X, Y, X_train, X_test, y_train, y_test, geneIDList, DescList, cl, selTicker):
    if selTicker == "vth":
        a, b, currentAUCList, X_trains, X_tests,y_score, sel, classifier= roc.findBestVarVarThreshold(X_train, X_test, y_train, y_test, cl)
    #elif selTicker == "uni":
    #elif selTicker == "pca":
    else:
        a, b, currentAUCList, X_trains, X_tests,y_score, sel, classifier= roc.findBestVarRestSel(X_train, X_test, y_train, y_test, cl, selTicker)
    return a, b, currentAUCList, X_trains, X_tests,y_score, sel, classifier
    
def intoDFFromStart(X, Y, X_train, X_test, y_train, y_test, geneIDList, DescList, cl, selTicker):
    #if selTicker == "vth":
    #    a, b, currentAUCList, X_trains, X_tests,y_score, sel, classifier= roc.findBestVarVarThreshold(X_train, X_test, y_train, y_test, cl)
    #elif selTicker == "uni":
    #elif selTicker == "pca":
    #else:
    #    a, b, currentAUCList, X_trains, X_tests,y_score, sel, classifier= roc.findBestVarRestSel(X_train, X_test, y_train, y_test, cl, selTicker)
        #if selTicker == "uni":
        #    sel = SelectKBest(score_func=chi2, k=a)
        #elif selTicker == "pca":
        #    sel = 
    
    a, b, currentAUCList, X_trains, X_tests,y_score, sel, classifier = intoPreFromStart(X, Y, X_train, X_test, y_train, y_test, geneIDList, DescList, cl, selTicker)
    df = feature_selection.totalOutputWithData(X_trains, X_tests, y_train, y_test, geneIDList, DescList, cl, sel, selTicker)
    #fig = plt.figure()
    #fig.set_size_inches(17.5, 9.5)
    #plt.plot(b * np.arange(len(currentAUCList)), 100*currentAUCList,color='navy', linestyle=':', linewidth=4)

    
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('Variance Threshold Finding Iteration')
    #plt.ylabel('Estimated AUC')
    #plt.title('Variance Threshold Maximizing AUC')
    return df, (b * np.arange(len(currentAUCList)), 100*currentAUCList)

def intoROCFromStart(X, Y, X_train, X_test, y_train, y_test, geneIDList, DescList, cl, selTicker):
    #if selTicker == "vth":
    #    a, b, currentAUCList, X_trains, X_tests,y_score, sel, classifier= roc.findBestVarVarThreshold(X_train, X_test, y_train, y_test, cl)

    #else:
    #    a, b, currentAUCList, X_trains, X_tests,y_score, sel, classifier= roc.findBestVarRestSel(X_train, X_test, y_train, y_test, cl, selTicker)
    
    a, b, currentAUCList, X_trains, X_tests,y_score, sel, classifier = intoPreFromStart(X, Y, X_train, X_test, y_train, y_test, geneIDList, DescList, cl, selTicker)
    fig = plt.figure()
    fig.set_size_inches(17.5, 9.5)
    plt.plot(np.arange(len(currentAUCList)), currentAUCList,
             color='navy', linestyle=':', linewidth=4)

    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Variance Threshold Finding Iteration')
    plt.ylabel('Estimated AUC')
    plt.title('Variance Threshold Maximizing AUC')    
    
    return fig
    
# if above works do fig = something   
  
#df = intoDFFromStart(X, Y, X_train, X_test, y_train, y_test, geneIDList, DescList, cl, "uni")
#print(df)
#roc =  intoROCFromStart(X, Y, X_train, X_test, y_train, y_test, geneIDList, DescList, cl, "uni")
        

###Mkae the above into fn and do return df
###Also try not to repeat computations in actual webapp and optimize