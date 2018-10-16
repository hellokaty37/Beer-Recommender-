"""
This is a file that will be called when we need to create a vector space
representing the semantic similiarity of all the words in a corpus.  For the
purposes of the beer recommendation project, the only terms we will end up
referencing for the recommender will be keywords, which is the bulk of the other
files in this project.

For this file, though, the word2vec model will be trained on the entire beer
review corpus (tokenized), which gives more context for the word2vec neural net
to play off of in creating the vector space.  Once this is created, the main
program will call the similarity function, referencing this vector space, to
check the average distance of keywords.
"""

#Import libraries
import gensim
import logging
import nltk
from nltk.tokenize import word_tokenize
import csv
import os
from os import path
from tempfile import mkstemp

csv_path = os.path.join(os.path.realpath('Project Data'),
'BeerAdvocate-DataNoAlterations.csv')

def load_data(csv_location):
    fullReviewCorpusList = []
    tokenizedReviewCorpusList = []
    print("start")

    with open(csv_location, 'r') as beerData:
        csvReader = csv.reader(beerData)
        #Get header titles from CSV file.
        header = csvReader.next()

        #Find indices of the beer name and review text, which are the fields
        #we care about for this application.
        reviewTextIndex = header.index("review_text")

        print("Creating text corpus.")
        for entry in csvReader:
            review = entry[reviewTextIndex]
            fullReviewCorpusList.append(review)
        length = len(fullReviewCorpusList)
        print("Tokenizing corpus of length ", length)
        for i in range(0, length):
            tokenizedReviewCorpusList.append(nltk.word_tokenize(fullReviewCorpusList[i]))
            print("Entry ", i, " of ", length, "tokenized.")
        print("Done creating text corpus.")
        return(tokenizedReviewCorpusList)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

fullReviewFile = load_data(csv_path)
print("Building Model")
model = gensim.models.Word2Vec(fullReviewFile, min_count=1, workers=5)

fs, temp_path = mkstemp("gensim_temp")
model.save(temp_path)
