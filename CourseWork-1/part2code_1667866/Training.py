
from urllib import request
from imports import*
from MLHelpers import DumpModel
from helpers import RequestValue
from Constants import variablesDict


#creats a pipe line based on model, feature and feature selector then fits the train sets into the the pipeline
#this function is used heavily and is essential for the programmes functionalities
def GenerateFittedPipeline(model, feature, featureSelector, xTrain, yTrain):
    pipe = Pipeline([("feature",feature), ("feature-Selector", featureSelector), ("model" ,model)])   
    pipe.fit(xTrain, yTrain)
    return pipe

    
#trains a model against a single features and dumps it into a joblib file
def TrainModel(modelName, model, featureName, feature, xTrain, yTrain, kFeatures):
    #creates a feature selectore object
    
    #fitted pipe line
    fittedPipe = GenerateFittedPipeline(model, feature, SelectKBest(chi2, k = kFeatures), xTrain, yTrain)
    #dump the trained pipe into a folder named by the model name and a file based on the feature and if 
    #it's a model from the cv operation, add the fold number into the a diffrent folder and add fold number into 
    #the file name
    folderName = "Normal" + "-" + modelName
    fileName =  folderName + "-" + featureName
    DumpModel(folderName, fileName, fittedPipe)
    
    

def TrainModels(modelsDict, featuresDict, xTrain, yTrain,kFeatures):
    for modelName, model in modelsDict.items():
            for featureName, featureTuple in featuresDict.items():
                TrainModel(modelName, model, featureName, featureTuple[0], xTrain, yTrain, kFeatures)