from imports import os, errno, string, stopwords, csv
#helper function to retrevie values from vars dict, it's done in this way to avoid changing the dict
def RequestValue(someDict,key):
    return someDict[key]


#creates a dir if not exist, deals with errors
def CreateDir(dirName):
    try:
        os.makedirs(dirName)
        print("Created ", dir)
    except OSError as e:
        
        if e.errno != errno.EEXIST:
            raise
            
            
#cleans any given string variable, from stopwords, punctioation, digits and extra spaces
def CleanString(textString):
    stop = set(stopwords.words("english"))
    punctuation= '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    textString = textString.strip()
    textString = " ".join([word for word in textString.split() if word not in stop])
    textString = textString.translate(str.maketrans('', '', string.punctuation))
    textString = textString.lower()
    textString = ''.join((element for element in textString if not element.isdigit()))
    newString=""
    for x in textString:
        if x not in punctuation:
            newString=newString+x
    return newString
    
#writes a txt file into a string, it also removes all whitespaces
def WriteTxtFileToString(txtFile, encodingType):
    with open(txtFile, 'r',encoding= encodingType) as file:
        data = file.read().replace('\n',' ')
        return data
    

#this function is reposnsible for populating a csv file with contents of multiple txt files in the structure provided by the university ('bbc folder')
def PopulateCsvFile(rootDir, csvFile, csvHeaderList, encodingType):
    CreateDir("CleanedData")
    with open(csvFile, "w", newline='',encoding=encodingType) as csvfile:
        #opens a writer object to populate headers in csv file
        writer = csv.DictWriter(csvfile, csvHeaderList)
        writer.writeheader()
        #flush
        writer = csv.writer(csvfile)

        
        #oswalk of target rootDir
        for path, dirs,files in os.walk(rootDir):
            #gets current dir in which os.walk is at
            currentDir = os.path.basename(os.path.normpath(path))
            #loops over the files of the current dir
            countOfFile = 0
            for file in sorted(files):
                #stores the path of the current file in the loop
                pathOfCurrentFile = os.path.join(path, file)
                #creats a csv row to be added
                string = WriteTxtFileToString(pathOfCurrentFile, encodingType)
                #cealns the string
                csvRow = [currentDir, CleanString(string) ]
                #print(os.path.join(path, file),csvRow)
                writer.writerow(csvRow)
                countOfFile += 1
                
            print("a total of ", countOfFile, " Articles in", path , " have been sucessfully read/write" )
