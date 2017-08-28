import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
try:
    from .feature_selection import VarianceThresholdFunction, FeatureImportance, PCAFunc, UnivFeatureFunction
except:
    from feature_selection import VarianceThresholdFunction, FeatureImportance, PCAFunc, UnivFeatureFunction

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation

# Initialize the MLP
num_classes = 4

def generate_results(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.show()
    print('AUC: %f' % roc_auc)
    
def generate_results_multiclass(y_test, y_score):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= num_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    #df1 = pd.DataFrame(fpr)
    
    #df = pd.DataFrame(d)

    #df.plot(style=['o','rx'])
    
    # Plot all ROC curves
#    fig = plt.figure()
#    fig.set_size_inches(17.5, 9.5)
#    plt.plot(fpr["micro"], tpr["micro"],
#             label='micro-average ROC curve (area = {0:0.2f})'
#                   ''.format(roc_auc["micro"]),
#             color='deeppink', linestyle=':', linewidth=4)
#    
#    plt.plot(fpr["macro"], tpr["macro"],
#             label='macro-average ROC curve (area = {0:0.2f})'
#                   ''.format(roc_auc["macro"]),
#             color='navy', linestyle=':', linewidth=4)
#    
#    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#    for i, color in zip(range(num_classes), colors):
#        plt.plot(fpr[i], tpr[i], color=color, 
#                 label='ROC curve of class {0} (area = {1:0.2f})'
#                 ''.format(i, roc_auc[i]))
#    
#    plt.plot([0, 1], [0, 1], 'k--')
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    #plt.title('ROC for Deep Neural Network')
#    plt.legend(loc="lower right")


#    return fig
    return [fpr, tpr, roc_auc]




def plotROC_SVMLinear(X_train, X_test, y_train, y_test ):
    y_train= label_binarize(y_train, classes=[0, 1, 2, 3])
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3])

    # Learn to predict each class against the other
    

    cl = svm.SVC(kernel='linear', probability=True)
    classifier = OneVsRestClassifier(cl)
    
    classifier.fit(X_train, y_train)
    
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    
    # Compute ROC curve and ROC area for each class
    #fig = generate_results_multiclass(y_test, y_score)
    #fig.title('ROC for Support Vecotr Machine: Linear Kernel')
    #fpr, tpr, _ = roc_curve(y_test, y_score)
    return generate_results_multiclass(y_test, y_score)

def plotROC_SVMGaussian(X_train, X_test, y_train, y_test ):
    y_train= label_binarize(y_train, classes=[0, 1, 2, 3])
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3])

    # Learn to predict each class against the other
    

    cl = svm.SVC(kernel='rbf', probability=True)
    classifier = OneVsRestClassifier(cl)
    
    classifier.fit(X_train, y_train)
    
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    
    # Compute ROC curve and ROC area for each class
    #fig = generate_results_multiclass(y_test, y_score)
    #plt.title('ROC for Support Vecotr Machine: RBF Kernel')
    #fpr, tpr, _ = roc_curve(y_test2, y_score)
    return generate_results_multiclass(y_test, y_score)

def plotROC_RFC(X_train, X_test, y_train, y_test ):
    y_train= label_binarize(y_train, classes=[0, 1, 2, 3])
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3])

    # Learn to predict each class against the other
    
    #cl = skflow.DNNClassifier(hidden_units=[13, 10,  5], n_classes=5)
    cl = RandomForestClassifier(n_estimators = 50)

    #cl = svm.SVC(kernel='rbf', probability=True)
    classifier = OneVsRestClassifier(cl)
    
    classifier.fit(X_train, y_train)
    #clf_probs = classifier.predict_proba(X_test)
    #y_score = log_loss(y_test, clf_probs)
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    #y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    
    # Compute ROC curve and ROC area for each class
    #fig = generate_results_multiclass(y_test, y_score)
    #plt.title('ROC for Random Forest Classifier')
    fpr, tpr, _ = roc_curve(y_test, y_score)
    return (fpr,tpr)

#CL is something like RandomForestClassifier(n_estimators = 50)
#Sel is something like VarianceThreshold(1.0) 같은 것
# So when the user inputs CL and Sel, they have to be adjusted like that
def plotROC_CL_SEL(X_train, X_test, y_train, y_test, cl, sel):
    y_train= label_binarize(y_train, classes=[0, 1, 2, 3])
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3])

    # Learn to predict each class against the other
    
    #cl = skflow.DNNClassifier(hidden_units=[13, 10,  5], n_classes=5)

    #cl = svm.SVC(kernel='rbf', probability=True)
    classifier = OneVsRestClassifier(cl)
    
    classifier.fit(X_train, y_train)
    #clf_probs = classifier.predict_proba(X_test)
    #y_score = log_loss(y_test, clf_probs)
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    #y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    
    # Compute ROC curve and ROC area for each class
    #fig = generate_results_multiclass(y_test, y_score)
    #plt.title('ROC for clName selName')
    fpr, tpr, _ = roc_curve(y_test, y_score)
    return (fpr,tpr)

#from sklearn.feature_selection import VarianceThreshold
#cl = RandomForestClassifier(n_estimators = 50)
#sel = sel = VarianceThreshold(1.0)
#plotROC_CL_SEL(X_train, X_test, y_train, y_test, cl, sel)

def estimateROCGivenClassifier(y_test, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])  
    #1 - E[f(pos)] E[1 - f(neg)]
    return 1- np.mean(fpr["micro"]) * np.mean(1-tpr["micro"])


##Do gradient Descent over varience threshold

#cl = svm.SVC(kernel='linear', probability=True)

def findBestVarVarThreshold(X_train, X_test, y_train, y_test, cl):
    y_train= label_binarize(y_train, classes=[0, 1, 2, 3])
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3])
    
    # Learn to predict each class against the other
    thresholdn = 0.0
    #Pseudo-gradient Descent until AUCestimator maximized
    prevAUC = 0.0
    currentAUC = 0.0000000000000000000000000000000000000000000000000001
    meanSoFar = 0.0000000000000000000000000000000000000000000000000001
    BatchSize = 0.001
    i = 0
    currentAUCList = []
    #print("yo")
    #Should be 100 but just do 20 for now
    while i<20 :
        #print(prevAUC)
        #print("hey!")
        prevAUC = currentAUC
        X_trains, X_tests, idxs1, sel = VarianceThresholdFunction(X_train, X_test, thresholdn)
        classifier = OneVsRestClassifier(cl)   
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
        currentAUC = estimateROCGivenClassifier(y_test, y_score)
        print(currentAUC)
        #print(currentAUC)
        thresholdn += BatchSize
        currentAUCList.append(currentAUC)
        i +=1
        meanSoFar = (meanSoFar * (i-1) + currentAUC) / i 
    currentAUCList= np.array(currentAUCList)
    maxEl = np.argmax(currentAUCList)
    X_trains, X_tests, idxs1, sel = VarianceThresholdFunction(X_train, X_test, 0.005* maxEl)
    classifier = OneVsRestClassifier(cl) 
        #print(X_trains.shape)
        #print(X_tests.shape)
        #print(X_train.shape)
        #print(y_train.shape)
    y_score = classifier.fit(X_trains, y_train).predict_proba(X_tests)       
    
    return  BatchSize * maxEl, BatchSize, currentAUCList, y_score, X_trains, X_tests, sel, classifier
    #return currentAUCList

#a= findBestVarVarThreshold(X_train, X_test, y_train, y_test, cl)
def findBestVarRestSel(X_train, X_test, y_train, y_test, cl, selTicker):
    y_train= label_binarize(y_train, classes=[0, 1, 2, 3])
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3])
    y_score = 0.0
    
    
    # Learn to predict each class against the other
    thresholdn = 0.0
    #Pseudo-gradient Descent until AUCestimator maximized
    prevAUC = 0.0
    currentAUC = 0.0000000000000000000000000000000000000000000000000001
    meanSoFar = 0.0000000000000000000000000000000000000000000000000001
    BatchSize = 5
    i = 1
    currentAUCList = []
    #print("yo")
    #Should be 100 but just do 20 for now
    #while i<= 50 and (prevAUC ==0.0 or meanSoFar > 0.5) :
    while i<= 50:
        #print(prevAUC)
        #print("hey!")
        prevAUC = currentAUC
        
        if selTicker == "uni":
            X_trains, X_tests, c, d, e, sel = UnivFeatureFunction(np.abs(X_train), np.abs(X_test), y_train, y_test, i)
        elif selTicker == "pca":
            X_trains, X_tests, sel = PCAFunc(X_train, X_test,y_train, y_test, i)
        #elif selTicker == "fim":
        #    X_trains, X_tests, c  = FeatureImportance(X_train, X_test,y_train, y_test, i)
            
        classifier = OneVsRestClassifier(cl) 
        #print(X_trains.shape)
        #print(X_tests.shape)
        #print(X_train.shape)
        #print(y_train.shape)
        y_score = classifier.fit(X_trains, y_train).predict_proba(X_tests)
        currentAUC = estimateROCGivenClassifier(y_test, y_score)
        print(currentAUC)
        #print(currentAUC)
        thresholdn += BatchSize
        currentAUCList.append(currentAUC)
        i +=1
        meanSoFar = (meanSoFar * (i-1) + currentAUC) / i 
        #if selTicker == "fim":
        #    break
    currentAUCList= np.array(currentAUCList)
    
    maxEl = np.argmax(currentAUCList)
    if selTicker == "uni":
        X_trains, X_tests, c, d, e, sel = UnivFeatureFunction(np.abs(X_train), np.abs(X_test), y_train, y_test, maxEl+1)
    elif selTicker == "pca":
        X_trains, X_tests, sel = PCAFunc(X_train, X_test,y_train, y_test, maxEl+1)
    classifier = OneVsRestClassifier(cl) 
        #print(X_trains.shape)
        #print(X_tests.shape)
        #print(X_train.shape)
        #print(y_train.shape)
    y_score = classifier.fit(X_trains, y_train).predict_proba(X_tests)       
        
    return  BatchSize * maxEl, BatchSize, currentAUCList, X_trains, X_tests, y_score, sel, classifier


#cl = RandomForestClassifier(n_estimators = 193)
#a, b = findBestVarRestSel(X_train, X_test, y_train, y_test, cl, "fim")
#a, b, c, d = findBestVarVarThreshold(X_train, X_test, y_train, y_test, cl)
def plotMaximizeCL_VarianceThreshold(X_train, X_test, y_train, y_test, cl):
    y_train= label_binarize(y_train, classes=[0, 1, 2, 3])
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3])
    
    # Learn to predict each class against the other
    thresholdn = 0.9
    #Pseudo-gradient Descent until AUCestimator maximized
    prevAUC = 0.0
    currentAUC = 0.0 
    BatchSize = 0.1
    i = 0
    currentAUCList = []
    while i<300 and prevAUC < currentAUC:
        prevAUC = currentAUC
        X_trains, X_tests = VarianceThresholdFunction(X_train, X_test, thresholdn)
        classifier = OneVsRestClassifier(cl)   
        y_score = classifier.fit(X_test, y_train).predict_proba(X_test)
        currentAUC = estimateROCGivenClassifier(y_test, y_score)        
        thresholdn += BatchSize
        currentAUCList.append(currentAUC)
        i +=1
    currentAUCList= np.array(currentAUCList )
    # Plot all ROC curves
    fig = plt.figure()
    fig.set_size_inches(17.5, 9.5)
    plt.plot(np.arange(i), currentAUCList,
             color='navy', linestyle=':', linewidth=4)

    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Variance Threshold Finding Iteration')
    plt.ylabel('Estimated AUC')
    plt.title('Variance Threshold Maximizing AUC')    
    
    return fig, np.argmax(currentAUCList), np.max(currentAUCList)


def plotMaximizeCL_OtherFeatureSelection(X_train, X_test, y_train, y_test, cl, func):
    y_train= label_binarize(y_train, classes=[0, 1, 2, 3])
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3])
    
    # Learn to predict each class against the other
    startn = 5
    #Pseudo-gradient Descent until AUCestimator maximized
    prevAUC = 0.0
    currentAUC = 0.0 
    BatchSize = 1
    i = 0
    currentAUCList = []
    while i<300 and prevAUC < currentAUC:
        prevAUC = currentAUC
        X_trains, X_tests = func(X_train, X_test, y_train, y_test, startn)
        classifier = OneVsRestClassifier(cl)   
        y_score = classifier.fit(X_test, y_train).predict_proba(X_test)
        currentAUC = estimateROCGivenClassifier(y_test, y_score)        
        startn += BatchSize
        currentAUCList.append(currentAUC)
        i +=1
    currentAUCList= np.array(currentAUCList )
    # Plot all ROC curves
    #fig = plt.figure()
    #fig.set_size_inches(17.5, 9.5)
    #plt.plot(np.arange(i), currentAUCList,
    #        color='navy', linestyle=':', linewidth=4)

    
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('Variance Threshold Finding Iteration')
    #plt.ylabel('Estimated AUC')
    #plt.title('Variance Threshold Maximizing AUC')    
    
    return np.argmax(currentAUCList), np.max(currentAUCList)


def mainPlotFunc(X_train, X_test, y_train, y_test):
    #Initializing model
    n_components = X_train.shape[1]
    if n_components <= 15:
        model = initialize_nn(n_components)
    else:
        model = initialize_nnAll(n_components)
    
    #Training model
    model.fit(X_train, y_train,
              batch_size=32, nb_epoch=100,
              verbose=0, callbacks=[],
              validation_data=None,
              shuffle=True,
              class_weight=None,
              sample_weight=None)
    
    y_score = model.predict(X_test)
    fig = generate_results_multiclass(y_test, y_score)
    plt.title('ROC for Deep Neural Network')
    return fig 


def initialize_nn(n_components):
    model = Sequential() # The Keras Sequential model is a linear stack of layers.
    model.add(Dense(20, init='uniform', input_dim=n_components)) # Dense layer
    model.add(Activation('tanh')) # Activation layer
    model.add(Dense(5, init='uniform')) # Another dense layer
    model.add(Activation('sigmoid')) # Another activation layer
    model.add(Dense(output_dim=num_classes))
    model.add(Activation("relu"))
    model.add(Activation('softmax')) # Softmax activation at the end
    sgd = SGD(lr=0.001, decay=1e-6, momentum=1, nesterov=True) # Using Nesterov momentum
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) # Using logloss
    return model

def initialize_nnAll(n_components):
    model = Sequential() # The Keras Sequential model is a linear stack of layers.
    model.add(Dense(1000, init='uniform', input_dim=n_components)) # Dense layer
    model.add(Activation('tanh')) # Activation layer
    model.add(Dense(20, init='uniform')) # Another dense layer
    model.add(Activation('sigmoid')) # Another activation layer
    model.add(Dense(output_dim=num_classes))
    model.add(Activation("relu"))
    model.add(Activation('softmax')) # Softmax activation at the end
    sgd = SGD(lr=0.001, decay=1e-6, momentum=1, nesterov=True) # Using Nesterov momentum
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) # Using logloss
    return model


