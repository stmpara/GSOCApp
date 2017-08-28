import csv 
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

###1. From CSV, make labels into list of strings
###2. From txt, make matrix of feature data
###3. Divide data into training and test data
def openCSVtoLabelList(file_name):  
    listDict = {} 
    with open(file_name, 'rU') as fh:
        csv_reader = csv.reader ( fh, delimiter = ',', dialect=csv.excel_tab )
        counter = 0
        listOfPhases = {0: ["I","IA", "IB"], 1: ["II","IIA", "IIB"], 2: ["III","IIIA", "IIIB"], 3: "IV"}
        for row in csv_reader:
            #Row 14 is phase data
            #Ditch script data at the end
            if counter !=0 and row[14]!= '':
                ##Want only ID and  Phase
                ##For convenience, convert phase to number
                ##(later needed for test train split)
                
                for phase in listOfPhases:
                    if row[14] in listOfPhases[phase]:
                        numLabel = phase
                listDict[row[1]] = numLabel
            counter +=1 
    return listDict

def printCSV(file_name):
        
    with open(file_name, 'rU') as fh:
        csv_reader = csv.reader ( fh, delimiter = ',', dialect=csv.excel_tab )
        for row in csv_reader:
            print(row)



##gets phase file(csv) as parameter and returns dictionary
def openCSVtoFeatureAllColumns(file_name):
    listDict = {}
    f = csv.reader(open(file_name))
    transposed = zip(*f)
    counter = 0
    #Column is row now
    IDList = []
    #IDList = transposed[0]
    DescriptionList = []
    for column in transposed:
        if counter ==0:
            IDList = column
        elif counter ==1:
            DescriptionList = column
        ##Omit the first two rows because 
        if counter >1 :
             ##if there exists 'NA', convert to NaN
             thisColumn = list(column[1:len(column)])
             #print(column)
             newColumn = []
             for elt in thisColumn:
                 ##Missing features -> 0.0 for now
                 if isinstance(elt, str) and ("NA" in elt):
                     #newColumn.append(float("NaN"))
                     newColumn.append(float(0.0))
                 elif isinstance(elt, str):
                     newColumn.append(float(elt))
                 else:
                     newColumn.append(elt)
             ##column[0] is ID
             listDict[column[0]] = newColumn
             #Sanity Check
             if len(newColumn) != len(thisColumn):
                 raise Exception("length not matched")
             
        counter += 1 
    #print(listDict['CL2005060332AA'])
    return listDict, IDList, DescriptionList
#a,b,c = openCSVtoFeatureAllColumns("/Volumes/Transcend/PythonPackgeGSOC/Datasets/geneAnalysis.csv")
#print(b)
#print(len(b))
##gets gene expression file(csv) as parameter and returns dictionary
def openCSVtoFeature(file_name):
    listDict = {}
    ##Mek1 and HER2 not in our file
    listOfGenes = ["AKT1", "ALK", "BRAF", "DDR2", "EGFR", "FGFR1", "HER2", "KRAS", "MEK1", "MET", "NRAS", "PIK3CA", "PTEN", "RET", "ROS1"]
    f = csv.reader(open(file_name))
    tbc = open('15_gene_file.csv', 'w')
    g = csv.writer(tbc)
    desc = listOfGenes
    ##First Select the 15 genes
    #gene_id_dict = {}
    counter = 0
    for row in f:
        if counter == 0:
            g.writerow(row)
        if row[0] in listOfGenes:
            g.writerow(row)    
        counter =1 
    tbc.close()
    gRead = csv.reader(open('15_gene_file.csv', 'r'))
    transposed = zip(*gRead)
    counter = 0
    #Column is row now
    for column in transposed:
        
        ##Omit the first two rows because 
        if counter >1 :
             ##if there exists 'NA', convert to NaN
             thisColumn = list(column[1:len(column)])
             newColumn = []
             for elt in thisColumn:
                 ##Missing features -> 0.0 for now
                 if isinstance(elt, str) and ("NA" in elt):
                     #newColumn.append(float("NaN"))
                     newColumn.append(float(0.0))
                 elif isinstance(elt, str):
                     newColumn.append(float(elt))
                 else:
                     newColumn.append(elt)
             ##column[0] is ID
             listDict[column[0]] = newColumn
             #Sanity Check
             if len(newColumn) != len(thisColumn):
                 raise Exception("length not matched")
             
        counter += 1 
    #print(listDict['CL2005060332AA'])
    return listDict, listOfGenes, desc

def sanityCheck(XDict, YDict, noIDS, noFeatures):
    XDictIDList = list(XDict.keys())
    if noIDS != len(YDict.keys()):
        raise Exception("Error with reading data: XDict and YDict have different number of IDs")
    for idOrder in range(noIDS):
        identity = XDictIDList[idOrder]
        if len(XDict[identity]) != noFeatures:
            print(XDict[identity])
            print(noFeatures)
            raise Exception("noFeatures not matched in XDict")


##Generates training set, test set, X, Y
def getData(file_name1, file_name2):
    ##First Divide Data into Test and Training
    ##Define X and Y from openCSV functions(numpy arrays)
    YDict = openCSVtoLabelList(file_name1)
    XDict, geneIDList, DescList = openCSVtoFeatureAllColumns(file_name2)
    IDOrderDict = {}
    ###X is feature matrix
    ###no of rows = no of ids
    ###no of columns = no of features(from checking .csv, we know all the id's have the same # of genes)
    XDictIDList = list(XDict.keys())
    noIDS = len(XDictIDList)
    firstID = XDictIDList[0]
    noFeatures =len(XDict[firstID])
    
    ##Do a sanity check on reading data for number of features and IDs
    sanityCheck(XDict, YDict, noIDS, noFeatures)
    
    X = np.zeros((noIDS, noFeatures))
    Y = np.zeros(noIDS)
    
    counter = 0
    for identity in YDict:
        ##Something wrong with reading data in this case
        if identity in IDOrderDict:
            raise Exception("Error with reading data:ID" + str(identity) + "overlapped")
        else:
            IDOrderDict[identity] = counter
            ##X[counter] is the row where all the genes appear
            X[counter, :] = XDict[identity]
            Y[counter] = YDict[identity]
            counter +=1 
    ##X and Y are numpy arrays from XDict and YDict
    ##They are ordered 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    return [X, Y, X_train, X_test, Y_train, Y_test,geneIDList, DescList]

def getDataAll(file_name1, file_name2):
    ##First Divide Data into Test and Training
    ##Define X and Y from openCSV functions(numpy arrays)
    YDict = openCSVtoLabelList(file_name1)
    XDict, geneIDList, DescList = openCSVtoFeatureAllColumns(file_name2)
    IDOrderDict = {}
    ###X is feature matrix
    ###no of rows = no of ids
    ###no of columns = no of features(from checking .csv, we know all the id's have the same # of genes)
    XDictIDList = list(XDict.keys())
    noIDS = len(XDictIDList)
    firstID = XDictIDList[0]
    noFeatures =len(XDict[firstID])
    
    ##Do a sanity check on reading data for number of features and IDs
    sanityCheck(XDict, YDict, noIDS, noFeatures)
    
    X = np.zeros((noIDS, noFeatures))
    Y = np.zeros(noIDS)
    
    counter = 0
    for identity in YDict:
        ##Something wrong with reading data in this case
        if identity in IDOrderDict:
            raise Exception("Error with reading data:ID" + str(identity) + "overlapped")
        else:
            IDOrderDict[identity] = counter
            ##X[counter] is the row where all the genes appear
            X[counter, :] = XDict[identity]
            Y[counter] = YDict[identity]
            counter +=1 
    ##X and Y are numpy arrays from XDict and YDict
    ##They are ordered 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    return [X, Y, X_train, X_test, Y_train, Y_test, geneIDList, DescList]


def intoDataFrame(X):
    columns=["AKT1", "ALK", "BRAF", "DDR2", "EGFR", "FGFR1", "KRAS", "MET", "NRAS", "PIK3CA", "PTEN", "RET", "ROS1"]
    primDict = {}
    for i in range(len(columns)):
        tempList = X[:,i]
        tempList = tempList.tolist()
        primDict[columns[i]] = tempList
    
    return pd.DataFrame(primDict)

import xlrd
#import nltk
#reload(sys)
#sys.setdefaultencoding('utf-8')

def csv_from_excel(excel_file):
    listOfFilesCreated = []
    workbook = xlrd.open_workbook(excel_file)
    all_worksheets = workbook.sheet_names()
    for worksheet_name in all_worksheets:
        worksheet = workbook.sheet_by_name(worksheet_name)
        your_csv_file = open(''.join([excel_file, '_', worksheet_name,'.csv']), 'w')
        wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

        for rownum in range(worksheet.nrows):
            #wr.writerow([unicode(entry).encode("utf-8") for entry in worksheet.row_values(rownum)])
            wr.writerow([entry for entry in worksheet.row_values(rownum)])
        your_csv_file.close()
        listOfFilesCreated.append(''.join([excel_file, '_', worksheet_name,'.csv']))
    #return listOfFilesCreated 
    
#X, Y, X_train, X_test, y_train, y_test = getData("/Volumes/Transcend/PythonPackgeGSOC/Datasets/LuncClinical.csv", "/Volumes/Transcend/PythonPackgeGSOC/Datasets/geneAnalysis.csv")
#a = pd.DataFrame.from_csv("/Volumes/Transcend/PythonPackgeGSOC/Datasets/LuncClinical.csv")