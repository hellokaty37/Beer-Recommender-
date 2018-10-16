"""
This part of the beer recommender program is meant to serve the purpose of
extracting keywords from the dataset.  It will use Scikit-Learn with a dataset
split into training data with identified keywords, test data with goal keywords,
and the rest of the dataset, which has no keywords identified ahead of time.
"""

#Import libraries
import nltk
from nltk.stem import WordNetLemmatizer
import numpy
import scipy
import sklearn
from sklearn.neural_network import MLPClassifier
#from sklearn import svm
#from sklearn.feature_extraction.text import CountVectorizer
import csv
import os
from os import path

def extractKeywords(csv_location, candidates, userReviewDict):
    #A candidate entry is a word, its frequency in the review, and its pos.
    y = checkTrainingKeywords(csv_location, candidates)
    X = getTrainingVectors(candidates)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X, y)

    testDict = userReviewDict
    userVectors = getTrainingVectors(testDict)
    prediction = clf.predict(userVectors)

    userReviewList = []
    for beer in userReviewDict:
        for item in userReviewDict[beer]:
            userReviewList.append(item)

    keywords = []
    index = 0
    for item in prediction:
        if item == 1:
            keywords.append(userReviewList[index][0])
        index += 1
    print(keywords)

def getTrainingVectors(candidateDictionary):
    trainingSamples = []
    for beer in candidateDictionary:
        for item in candidateDictionary[beer]:
            x = []
            x.append(item[1])
            x.append(item[3])
            trainingSamples.append(x)
    return(trainingSamples)


def checkTrainingKeywords(csv_location, candidateDictionary):
    keywordsDictionary = {}
    targetValues = []

    with open(csv_location, 'r') as beerData:
        csvReader = csv.reader(beerData)
        #Get header titles from CSV file.
        header = csvReader.next()

        #Find indices of the training keywords.
        beerNameIndex = header.index("beer_name")
        keywordIndex = header.index("keywords")

        #Create keywords dictionary of beers and their identified keywords.
        for entry in csvReader:
            name = entry[beerNameIndex]
            keywords = entry[keywordIndex]

            #Create keywords list for that beer's identified keywords.
            keywordsList = keywords.split(', ')
            """
            #Ensure keywords are lemmatized to match the candidate terms.
            for item in keywordsList:
                index = keywordsList.index(item)
                lemmatizer = WordNetLemmatizer()
                newItem = str(lemmatizer.lemmatize(item))
                keywordsList[index] = newItem
            """

            #Save keywords list in a dictionary to that beer with name as key.
            keywordsDictionary[name] = keywordsList

        for item in candidateDictionary:
            if item in keywordsDictionary:
                length = len(candidateDictionary[item])
                for i in range(0, length):
                    if candidateDictionary[item][i][0] in keywordsDictionary[item]:
                        #candidateDictionary[item][i].append(1)
                        targetValues.append(1)
                    else:
                        #candidateDictionary[item][i].append(0)
                        targetValues.append(0)
    return(targetValues)
