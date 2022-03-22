from helpers import RequestValue, CleanString
from imports import *
from Constants import variablesDict
from MLHelpers import CalculateScores


def EvaluateModels(xValid, yValid, type):
    evalScores = list()
    for root, dirs, files in os.walk(RequestValue(variablesDict, "modelsPath")):
        for file in files:
            if file.lower().endswith(".joblib".lower()) and type in file.lower():
               #loads a pipeline in dir
                newPipe = load(os.path.join(root, file))
                #predicts scores
                predict = newPipe.predict(xValid)
                accuracy, f1, recall, precision = CalculateScores(predict, yValid)
                print(file, " has been evaluated against data.")
                token = file.split('-')  
                evalScores.append({"ModelName":token[0] + token[1], "FeatureName": token[2].split('.')[0], "accuracy":accuracy, "f1":f1, "precision":precision, "recall":recall})
         
    df = pd.DataFrame(evalScores)            
    return df

def EvaluateString(string):
    string = CleanString(string)
    for root, dirs, files in os.walk(RequestValue(variablesDict, "modelsPath")):
        for file in files:
            if file.lower().endswith(".joblib".lower()):
                #loads a pipeline in dir
                newPipe = load(os.path.join(root, file))
                #predicts scores
                predict = newPipe.predict([string])
                print(file, " predicted:", predict)