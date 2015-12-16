'''
Created on Nov 17, 2015

@author: karthik
'''

import yaml
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet
import sys

wordnetDict = {'NN':wordnet.NOUN,'JJ':wordnet.ADJ,'VB':wordnet.VERB,'RB':wordnet.ADV}
lemma_cache = {}

def lemmatize(reviews, outFile):
  lemmatizer = WordNetLemmatizer()
  i = 0
  for line in reviews: # use yaml to avoid annoying unicode conversion with json
    i += 1
    print "Processing review", i, "\r", 
    review = yaml.safe_load(line)
    pos = nltk.pos_tag(nltk.word_tokenize(review['text']))
    pos = convertToWordNetTags(pos)
    lemmatizedText = ""
    for word in pos:
      if word in lemma_cache:
        lemmatizedText += lemma_cache[word[0]] + " "
      else:
        lemma = lemmatizer.lemmatize(word[0], word[1])
        lemma_cache[word[0]] = lemma
        lemmatizedText += lemma + " "
    
    review['text'] = lemmatizedText
    print >> outFile, json.dumps(review)
    if i % 100 == 0:
      outFile.flush()

    print "\n"

def doStemming(reviews, outFile):
  stemmer = SnowballStemmer("english", ignore_stopwords=True)
  i = 0
  for line in reviews: # use yaml to avoid annoying unicode conversion with json
    i += 1
    print "Processing review", i, "\r", 
    review = yaml.safe_load(line)
    pos = nltk.word_tokenize(review['text'])      
    stemmedText = ""
    for word in pos:
      if word in lemma_cache:
        stemmedText += lemma_cache[word] + " "
      else:
        stem = stemmer.stem(word)
        lemma_cache[word] = stem
        stemmedText += stem + " "
    
    review['text'] = stemmedText
    print >> outFile, json.dumps(review)
    if i % 100 == 0:
      outFile.flush()

  print "\n"

def convertToWordNetTags(POStags):
  # convert from Treebank tags to WordNet Tags prior to lemmatization  
  for i in xrange(len(POStags)):  
    if POStags[i][1][:2] in wordnetDict:
      POStags[i] = (POStags[i][0], wordnetDict[POStags[i][1][:2]])
    else:
      POStags[i] = (POStags[i][0], 'n')
    
  return POStags 
    
if __name__ == '__main__':
  
  reviewFile = sys.argv[1]
  outputFile = sys.argv[2]
  stem = sys.argv[3]
  
  with open(reviewFile, 'r') as reviews:
    with open(outputFile, 'w') as outFile:
      if stem.lower() == "true":
        print "Stemming text"
        doStemming(reviews, outFile)
      else:
        lemmatize(reviews, outFile)
    