"""
This is a module to take the reviews from the Beer Advocate dataset, send
them through the ReviewCleaner module, then use a keyword extractor on them.
At the end of this program, the results are saved out for use in the
BeerRecommender module, which is the main program in this package.
"""
#Import the natural language toolkit and the CSV package to parse data.
import nltk
from nltk.stem import WordNetLemmatizer
import csv
import os
from os import path

#Import keyword extraction function.
from BeerKeywordExtractor import extractKeywords
from ReviewCleaner import cleanReview, getTokens, cleanUpTokens, getUniques, getPos

"""
Function to load the beer names and individual reviews from the CSV and convert
them into a dictionary of beer names as keys with concatenated review corpora
as values.
"""
def load_data(csv_location):
    #Create a dictionary to hold beer names and review corpora.
    dataDict = {}
    beerDataList = []

    with open(csv_location, 'r') as beerData:
        csvReader = csv.reader(beerData)
        #Get header titles from CSV file.
        header = csvReader.next()

        #Find indices of the beer name and review text, which are the fields
        #we care about for this application.
        beerNameIndex = header.index("beer_name")
        reviewTextIndex = header.index("review_text")

        for entry in csvReader:
            name = entry[beerNameIndex]
            review = entry[reviewTextIndex]
            beerDataList.append([name, review])

        #Populate data dictionary with beer names and concatenated corpora.
        for item in beerDataList:
            if item[0] in dataDict:
                dataDict[item[0]] = dataDict[item[0]] + " " + item[1]
            else:
                dataDict[item[0]] = item[1]

        return(dataDict)

def processReviews():
    #Load the csv into the function that will read it and return the data dictionary.
    csv_path = os.path.join(os.path.realpath('Project Data'),
    'BeerAdvocate-DataTrainingData.csv')
    reviewDictionary = load_data(csv_path)

    #Clean up dictionary for keyword extraction.
    posTokensDictionary = cleanReview(reviewDictionary)
    print(posTokensDictionary)

    return(csv_path, posTokensDictionary)
