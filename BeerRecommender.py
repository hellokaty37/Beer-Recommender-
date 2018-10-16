"""
This is a program to recommend beers to a user given a beer that the user liked
and a description about why the user liked the beer.
It makes use of the Beer Advocate reviews dataset from
https://data.world/petergensler/beer-advocate-reviews/workspace/file?filename=BeerAdvocate-000.csv
"""

#Import keyword extraction function.
from BeerKeywordExtractor import extractKeywords
from QueryWord2Vec import queryVectorSpace
from ReviewCleaner import cleanReview, getTokens, cleanUpTokens, getUniques, getPos
from ProcessReviewsDataset import processReviews
"""
#Get beer review from user.
print("Please enter the name of a beer you enjoyed: ")
userBeerName = str(raw_input())
print("Enter a review of ~150 characters describing the beer you liked: ")
userBeerReview = str(raw_input())

#Save user review into a dictionary and clean it up for keyword extraction.
userBeerDict = {}
userBeerDict[userBeerName] = userBeerReview
posUserTokens = cleanReview(userBeerDict)
"""

#Initialize keywords model.
csv, reviewsDict = processReviews()
"""
#Get keywords from user review.
userBeerKeywords = extractKeywords(csv, reviewsDict, posUserTokens)
"""

#Compare words from two beers in vector space.
#similarityMeasure = queryVectorSpace(keywords)
