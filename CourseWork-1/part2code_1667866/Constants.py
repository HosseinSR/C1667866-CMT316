from asyncio.windows_events import NULL
from imports import*

#a dict struct for init variables in the programme
variablesDict = {
    "cwd": os.getcwd(),
    "targetCsvFile" :os.getcwd() +'/CleanedData' + "/processedData.csv",
    "fieldNames" : ["Category", "Text"],
    "rawDataPath": os.getcwd() + '/datasets_coursework1/bbc',
    "modelsPath": os.getcwd() + '/Models',
    "resultsPath":os.getcwd() + '/Results',
    "testSize": 0.2,
    "randomSeed": 4444,
    "encodingType":"unicode_escape",
    "kFeatures" : 250,
    "kFolds" : 3
    
}


modeslDictionary = {
    "NaiveBayes" : naive_bayes.MultinomialNB(),  
    "LogisticRegression": linear_model.LogisticRegression(max_iter=100),
    "SVM" : svm.SVC()
}
nonOpFeaturesDictionary = {
    "CountVectorizer": (CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features =  5000), NULL),
    
    
    "TFDIFUnigram": (TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',ngram_range = (1,2), max_features =  5000), NULL),
    
    
    
    "TFDIFChar":  (TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}',ngram_range = (1,2), max_features =  5000),NULL)
    }


featuresDictionary= {
    "CountVectorizer": ( CountVectorizer(analyzer='word', token_pattern=r'\w{1,}'), {"CountVectorizer__ngram_range": [(1,1), (1,2)],
                                                                                     "CountVectorizer__max_features" : [5000, 6000,7000],
                                                                                     
                                                                                    'feature-Selector__k' : [100,200,250,300,350]} ),
    
    
    "TFDIFUnigram": (TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',ngram_range = (1,2)),    {
                                                                                     "TFDIFUnigram__max_features" : [5000, 6000,7000],
                                                                                    
                                                                                     
                                                                                   'feature-Selector__k' : [100,200,250,300,350]} ),
    
    
    
    "TFDIFChar":  (TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}',ngram_range = (1,2)),  {
                                                                                     "TFDIFChar__max_features" : [5000, 6000,7000],
                                                                                     
                                                                                     
                                                                               'feature-Selector__k' : [100,200,250,300,350]} )
}
