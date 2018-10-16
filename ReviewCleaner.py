"""
This is a module with many functions used to clean up a review of a beer,
either from the Beer Advocate database or from the user's entry.
"""
#Import the natural language toolkit and the CSV package to parse data.
import nltk
from nltk.stem import WordNetLemmatizer

def cleanReview(reviewsDictionary):
    tokenizedReview = getTokens(reviewsDictionary)
    cleanTokensReview = cleanUpTokens(tokenizedReview)
    uniqueCleanTokens = getUniques(cleanTokensReview)
    posTokens = getPos(uniqueCleanTokens)

    return(posTokens)

"""
This is a function to tokenize the review text corpus of each beer.
It returns a new dictionary with beer names as keys and tokenized review corpora
as values.
"""
def getTokens(dataDictionary):
    tokensDict = {}
    for item in dataDictionary:
        tokens = nltk.word_tokenize(dataDictionary[item])
        tokensDict[item] = tokens
    return tokensDict

"""
This is a function that cleans up a dictionary of tokens from beer reviews,
removing capitalization, punctuation, numbers.
"""
def cleanUpTokens(tokensDictionary):
    for item in tokensDictionary:
        for token in tokensDictionary[item][:]:
            #Remove punctuation
            if token in ['.', ',', "'", '"', ';', ':']:
                tokensDictionary[item].remove(token)

            #Remove numbers
            elif any(char.isdigit() for char in token)==True:
                tokensDictionary[item].remove(token)

            #Make everything lowercase and lemmatize
            else:
                index = tokensDictionary[item].index(token)
                if token.islower()==False:
                    token = token.lower()

                lemmatizer = WordNetLemmatizer()
                token = str(lemmatizer.lemmatize(token))
                tokensDictionary[item][index] = token

    return(tokensDictionary)

"""
This is a function to convert a list of tokens in a review corpus into a list of
unique tokens and frequency of each token.
"""
def getUniques(tokensDictionary):
    uniquesDict = {}
    for item in tokensDictionary:
        for token in tokensDictionary[item]:
            if item not in uniquesDict:
                uniquesDict[item] = []
                uniquesDict[item].append([token, 1])
            else:
                for tokenArray in uniquesDict[item]:
                    if token in tokenArray:
                        tokenFound = 'yes'
                        tokenIndex = uniquesDict[item].index(tokenArray)
                        break
                    else:
                        tokenFound = 'no'
                if tokenFound == 'yes':
                    currentCount = uniquesDict[item][tokenIndex][1]
                    newCount = currentCount + 1
                    uniquesDict[item][tokenIndex] = [token, newCount]
                else:
                    uniquesDict[item].append([token,1])

    return(uniquesDict)

"""
This is a function to get the part of speech for each token and add that as a
feature of the token in the tokens dictionary, in addition to frequency count.
"""
def getPos(tokensDictionary):
    posDictionary = {}
    tokensList = []
    for item in tokensDictionary:
        length = len(tokensDictionary[item])
        for i in range(0, length):
            tokensList.append(tokensDictionary[item][i][0])
    posList = []
    posList = nltk.pos_tag(tokensList)
    counter = 0
    for item in tokensDictionary:
        length = len(tokensDictionary[item])
        for i in range(0, length):
            token = tokensDictionary[item][i][0]
            count = tokensDictionary[item][i][1]
            pos = posList[counter][1]
            if pos in ('FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'RB', 'RBR', 'RBS'):
                posBool = 1
            else:
                posBool = 0
            counter += 1
            if i==0:
                posDictionary[item] = []
                posDictionary[item].append([token, count, pos, posBool])
            else:
                posDictionary[item].append([token, count, pos, posBool])
    return(posDictionary)
