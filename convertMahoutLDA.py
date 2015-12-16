
import sys
import os
import time
import re
import shutil

from gensim import corpora, models
from MahoutCorpus import MahoutCorpus

NON_WORD_REGEX = re.compile('(?:\W|_)+')

# Converts output from Mahout LDA topic_term dist and doc_topic dist to format usable by pyLDAvis 
# the dictionary for pyLDAvis is created from the topic term distribution, not the vocab.txt file
def getCategories():
  dirnames = [["health", "eventservices", "pets" ,"publicservicesgovt" ,"nightlife" ,"shopping"]]
  dirnames.append(["homeservices", "education", "localflavor", "arts", "beautysvc", "financialservices", "localservices"])
  dirnames.append(["food", "auto", "active", "hotelstravel", "religiousorgs", "massmedia", "professional"])
  dirnames.append(["restaurants"])
  
  dirnames = [item for sublist in dirnames for item in sublist]
  
#  dirnames = ["localflavor"]
  return dirnames

def printDict(dictionary):
  print dictionary.token2id
  

def createDict(pathToDir, topicTermDistFile, saveLoc = None):
  # we create the dictionary from the topic-term distributions file
  # to avoid issues Mahout seems to have with Unicode characters among other things
  print "Creating dictionary for", pathToDir
  start = time.clock()
  # every line in the topic-term distribution file contains all the terms for a given topic
  # so we only need to read one line to get all the terms
  line = topicTermDistFile.readline()
  line = line.strip()[1:-1] # strip out leading and trailing { and }
  
  vals = [x.split(':') for x in line.split(",")]
  # create a set of terms in this corpus that are represented in the topic-term distribution file
  vocab_set = set([val[0] for val in vals])
  dictionary = corpora.Dictionary([sorted(vocab_set)], prune_at=None)
  
  dictionary.compactify()
      
  if saveLoc is not None:
    dictionary.save(saveLoc)
  
  print "Time to create dictionary is {0:.3f} seconds".format(time.clock() - start)
    
  return dictionary  
def createCorpus(parent, directory, dictionary, saveLoc = None):
  review_corpus = MahoutCorpus(parent + directory + "/", NON_WORD_REGEX, dictionary)

  start = time.clock()
  review_tfidf = models.tfidfmodel.TfidfModel(review_corpus, normalize=True)
  review_corpus = review_tfidf[review_corpus]    
  print "Time to create tf-idf is {0:.3f} seconds".format(time.clock() - start)
  if saveLoc is not None:
    corpora.MmCorpus.serialize(saveLoc, review_corpus)
  
  return review_corpus

def combineTopicTermFiles(distDir, outputFilename, numTopics):
  with open(outputFilename, 'w') as f:
    for i in xrange(numTopics):
      with open(distDir + 'topicterm_dist' + str(i) + '.json', 'r') as infile:
        for line in infile:
            print >> f, line, 
  
        
if __name__ == '__main__':    
  print "\nUsage: python convertMahoutLDA <high/low> <directory> <numTopic>"
  print "If the last two are left out, it runs on all categories and topics\n\n"
  
  if sys.argv[1].lower() == "high":    
    reviewType = "high_reviews"
  else:
    reviewType = "low_reviews"
    
  corpusParent = "../train/" + reviewType + "/"
  if len(sys.argv) > 2:
    directories = [sys.argv[2]]
    numTopics = int(sys.argv[3])
  else:
    directories = getCategories()
    numTopics = -1
    
  if numTopics == -1:
    numTopicChoices = [3, 5, 10]
  else:
    numTopicChoices = [numTopics]
  
  for directory in directories:
    outputTarget = "../processed/" + reviewType + "_mahoutLDA/" + directory
    files = os.listdir(corpusParent + directory)
    
    topicTermFile = "../mahout_results/reviews/" + reviewType + "/" + directory + "/3/model/topicterm_dist0.json"
    with open(topicTermFile) as topicTermDistFile:   
      dictionary = createDict(corpusParent + directory + "/", topicTermDistFile, saveLoc = outputTarget + ".dict")
    
    review_corpus = createCorpus(corpusParent, directory, dictionary, saveLoc = outputTarget + ".mm")
  
    for numTopics in numTopicChoices:
      distParent = "../mahout_results/reviews/" + reviewType + "/" + directory + "/" + str(numTopics) + "/model/"
      
      # copy doc-topic dist file 
      shutil.copy(distParent + 'doctopic_dist.json', outputTarget + '_' + str(numTopics) + '_doctopic_dist.json')
  
      combinedTopicTermDistFile = outputTarget + '_' + str(numTopics) + '_topicterm_dist.json'
      combineTopicTermFiles(distParent, combinedTopicTermDistFile, numTopics)
    
  
    
