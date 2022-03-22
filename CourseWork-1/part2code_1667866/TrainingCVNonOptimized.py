import imp
from tabnanny import verbose

from imports import*
from Constants import variablesDict
from MLHelpers import DumpModel
from helpers import RequestValue
from statistics import mean


def GenerateNonOpPipe(modelName, model, featureName, featuerTuple, xTrain, yTrain, kFeatures):
    scoring = {'accuracy': make_scorer(accuracy_score),'recall': make_scorer(recall_score, average = 'macro'),
               'precision': make_scorer(precision_score, average = 'macro')
               ,'f1': make_scorer(f1_score, average = 'macro')}

    pipe = Pipeline([(featureName,featuerTuple[0]), ("feature-Selector", SelectKBest(chi2, k = kFeatures)), (modelName ,model)])
    clf = cross_validate(pipe, xTrain, yTrain, cv=RequestValue(variablesDict, "kFolds" ), scoring= scoring, n_jobs = -1 ,return_train_score=True, verbose = 1)
    #print(clf)

    
    trainAccuracy= mean(clf['train_accuracy'])
    trainF1=mean (clf['train_f1'])
    trainPrecision=mean(clf['train_precision'])
    trainRecall=mean(clf['train_recall'])

    trainScores = {"modelName": "Train-" + modelName, "featureName": featureName, "AVGaccuracy" :trainAccuracy , "AVGf1": trainF1, "AVGprecision":trainPrecision , "AVGrecall":trainRecall }

    testAccuracy= mean(clf['test_accuracy'])
    testF1 =mean (clf['test_f1'])
    testPrecision =mean(clf['test_precision'])
    testRecall =mean(clf['test_recall'])

    testScores = {"modelName": "Test-"+modelName, "featureName": featureName, "AVGaccuracy" :testAccuracy , "AVGf1": testF1, "AVGprecision":testPrecision , "AVGrecall":testRecall }

    return trainScores, testScores

def TrainModelsOnCvNonOptimized(modelsDict,featuresDict, dataFrame, kFeatures):
    # a list to store test scorse when cross validation operation is in place#predicting xvalid and yvalid folds
    allTrainScores = list()
    allTestScores = list()
    
    xVal = dataFrame["Text"]
    yVal = dataFrame["Category"]

    for modelName, model in modelsDict.items():
            for featureName, featuerTuple in featuresDict.items():
                trainScores, testScores = GenerateNonOpPipe(modelName, model, featureName, featuerTuple, xVal, yVal, kFeatures)
                allTrainScores.append(trainScores)
                allTestScores.append(testScores)

    allTrainScoresdf = pd.DataFrame(allTrainScores)
    allTestScoresdf = pd.DataFrame(allTestScores)
    return allTrainScoresdf, allTestScoresdf