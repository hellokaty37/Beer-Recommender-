"""
This is a program to query the pre-made Word2Vec vector space in the file
Word2Vec.py
"""
#Import libraries
import gensim
import logging
from tempfile import mkstemp

def queryVectorSpace(beersDict, beer1, beer2):
    beer1Words = beersDict[beer1]
    beer2Words = beersDict[beer2]
    model = gensim.models.Word2Vec.load('/var/folders/j4/r_52jvl903z7l7rs3hl0j7fm4l18m_/T/tmpTeiw4Agensim_temp')
    return(model.similarity(beer1Words, beer2Words))
