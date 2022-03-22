from urllib import request
from imports import*
from MLHelpers import DumpModel
from helpers import RequestValue
from Constants import variablesDict

#creats a pipe line based on model, feature and feature selector then fits the train sets into the the pipeline
#this function is used heavily and is essential for the programmes functionalities
def GenerateCVPipeline(modelName,model, featureName, featureTuple, xTrain, yTrain):
    #inits scoring metrics
    scoring = {'accuracy': make_scorer(accuracy_score),'recall': make_scorer(recall_score, average = 'macro'),
               'precision': make_scorer(precision_score, average = 'macro')
               ,'f1': make_scorer(f1_score, average = 'macro')}

    #creates a pipe line based on give features and models  
    pipe = Pipeline([(featureName,featureTuple[0]), ("feature-Selector", SelectKBest(chi2)), (modelName ,model)])
    #passed a val for parms from feature tuple stored in constants file
    parameters = featureTuple[1]
    #grid search for finding best parms
    clf = GridSearchCV(pipe, parameters,verbose=1,cv =RequestValue(variablesDict, "kFolds"), scoring= scoring, refit ="accuracy", n_jobs=-1, return_train_score=True)
    #fits models against search space
    clf.fit(xTrain, yTrain)
    #from stack
    #https://datascience.stackexchange.com/questions/64447/what-is-my-training-score-the-mean-train-score-or-mean-test-score
    #The mean_test_score is actually the mean score of the validation step for each fold. The "test" word is probably not well chosen by sklearn in that case, if you want to make the distinction between validation and test.
    #get best scores

    
    trainAccuracy= clf.cv_results_['mean_train_accuracy'][clf.best_index_]
    trainF1=clf.cv_results_['mean_train_f1'][clf.best_index_]
    trainPrecision=clf.cv_results_['mean_train_precision'][clf.best_index_]
    trainRecall=clf.cv_results_['mean_train_recall'][clf.best_index_]

    trainScores = {"modelName": "Train-" + modelName, "featureName": featureName, "accuracy" :trainAccuracy , "f1": trainF1, "precision":trainPrecision , "recall":trainRecall }

    testAccuracy= clf.cv_results_['mean_test_accuracy'][clf.best_index_]
    testF1 =clf.cv_results_['mean_test_f1'][clf.best_index_]
    testPrecision =clf.cv_results_['mean_test_precision'][clf.best_index_]
    testRecall =clf.cv_results_['mean_test_recall'][clf.best_index_]


    testScores = {"modelName": "Train-" + modelName, "featureName": featureName, "accuracy" :testAccuracy , "f1": testF1, "precision":testPrecision , "recall":testRecall }
    print("Best model parms:")
    print(clf.best_estimator_)
    #returns the fitted classifier/gridsearchcv object with corssvalidation operation
    #retruns a dict of mean results from cross-validation operation
    return clf.best_estimator_, trainScores, testScores


#trains several models on cross-validation utilizing optimized grid search
#returns a dataframe of the train scores of the cross-validation processes 
def TrainModelsCVOptimized(modelsDict, featuresDict, xTrain, yTrain):
    #a list to store retruned dicts from GenerateCVPipeline to latter create a dataframe from the information
    allTrainScores = list()
    allTestScores = list()
    #for each model to be trained
    for modelName, model in modelsDict.items():
            #for each feature we want to train our models at
            for featureName, featureTuple in featuresDict.items():
                #generated a fitted optimized GridSearchCV object and store it in var[clf] 
                #get results of training, mean[f1,acc, recall, prescion] and store in var[modelFeatureTrainScore]
                clf, trainScores, testScores= GenerateCVPipeline(modelName,model, featureName, featureTuple, xTrain, yTrain)
                #need train info
                #dump model
                folderName = "OptimizedCV" + "-" + modelName
                fileName = folderName + "-" + featureName
                DumpModel(folderName, fileName, clf)
                allTrainScores.append(trainScores)
                allTestScores.append(testScores)

                
    allTrainScoresdf = pd.DataFrame(allTrainScores)
    allTestScoresdf = pd.DataFrame(allTestScores)
    return allTrainScoresdf, allTestScoresdf