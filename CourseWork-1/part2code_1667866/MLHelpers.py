from imports import*
from helpers import*
from Constants import variablesDict


#creates a dataframe of desired csv file 
def InitDataFrame(CsvFile, encodingType):
    dataFrame = pd.read_csv(CsvFile, encoding= encodingType)
    return dataFrame

#splits dataframe based on test size 
def SplitDataFrame(dataFrame, testSize, randomSeed):  
    xTrain, xValid, yTrain, yValid = train_test_split(dataFrame[RequestValue(variablesDict,"fieldNames" )[1]], dataFrame[RequestValue(variablesDict,"fieldNames" )[0]],
     test_size=testSize, random_state=randomSeed)
    return xTrain, xValid, yTrain, yValid



def DumpModel(folderName, fileName, fittedPipe):
    #check if the is first call to this function
    #if this is the case a models dir will be created for all models to be stored at 
    modelsDir = os.path.join(RequestValue(variablesDict, "cwd"), "Models")
    CreateDir(modelsDir)
    folderDir = os.path.join(modelsDir, folderName)  
    CreateDir(folderDir)
    filePath = os.path.join(folderDir, fileName) + ".joblib"
    dump(fittedPipe, filePath)


def CalculateScores(predictions, yValid):
    accuracy = metrics.accuracy_score(predictions, yValid)
    f1 = metrics.f1_score(predictions, yValid, average = "macro")
    recall = metrics.recall_score(predictions, yValid, average = "macro")
    precision = metrics.precision_score(predictions, yValid, average = "macro")
    return accuracy, f1, recall, precision


def ExcelAndShowCase(dataFrame, fileName):
    file = RequestValue(variablesDict, "resultsPath") + fileName + ".xlsx"
    dataFrame.to_excel(file)
    return dataFrame

