'''
Created on Nov 25, 2015

@author: karthik
'''
import logging
import os
import re
import sys 
import time

from gensim import corpora, models
from nltk.corpus import stopwords

from MyCorpus import MyCorpus
from gensim.models.hdpmodel import HdpModel


MIN_DOCS = 5
MAXDF = 0.75
NON_WORD_REGEX = re.compile('(?:\W|_|\d)+')

def printDict(dictionary):
  print dictionary.token2id
  
def createDict(pathToDir, files, saveLoc = None, debugCalcDF = False):
  print "Creating dictionary for", pathToDir
  start = time.clock()
  dictionary = corpora.Dictionary([])
  docFreqs = {}
  nltk_stopwords = stopwords.words('english')
  
  for review in files:
    if not os.path.isfile(pathToDir + review): 
      continue
    with open(pathToDir + review, 'r') as f:
      contents = f.readline().decode("ascii", "ignore")
      
      words = NON_WORD_REGEX.split(contents)
      words = [word for word in words if word not in nltk_stopwords]
      
      dictionary.add_documents([words])
      if debugCalcDF:
        docFreq = {word:1 for word in words}
        for word in docFreq:
          if word not in docFreqs:
            docFreqs[word] = 0
          docFreqs[word] += 1
          
  dictionary.filter_extremes(MIN_DOCS, MAXDF)
  dictionary.compactify()
  
  if debugCalcDF:
    print docFreqs
    
  if saveLoc is not None:
    dictionary.save(saveLoc)
  
  print "Time to create dictionary is {0:.3f} seconds".format(time.clock() - start)
    
  return dictionary  
  
def createCorpus(parent, directory, dictionary, saveLoc = None):
  review_corpus = MyCorpus(parent + directory + "/", NON_WORD_REGEX, dictionary)

  start = time.clock()
  review_tfidf = models.tfidfmodel.TfidfModel(review_corpus, normalize=True)
  review_corpus = review_tfidf[review_corpus]    
  print "Time to create tf-idf is {0:.3f} seconds".format(time.clock() - start)
  if saveLoc is not None:
    corpora.MmCorpus.serialize(saveLoc, review_corpus)
  
  return review_corpus
  
def doAll(parent):
  dirnames = getCategories()
  
  numTopicsChoices = [3, 5, 10]
  for directory in dirnames:
    files = os.listdir(parent + directory)
    
    dictionary = createDict(parent + directory + "/", files, 
                            saveLoc = "../processed/" + parent[9:-1] + "_lda/" + directory + ".dict")
    
    review_corpus = createCorpus(parent, directory, dictionary, 
                 saveLoc = "../processed/" + parent[9:-1] + "_lda/" + directory + ".mm")
  
    for numTopics in numTopicsChoices:
      print "Processing", directory, "for", str(numTopics), "topics"
      start = time.clock()
      lda = models.ldamodel.LdaModel(corpus=review_corpus, id2word=dictionary, 
                                   num_topics=numTopics, chunksize=10)
      lda.save("../processed/" + parent[9:-1] + "_lda/" + directory + "_" + str(numTopics) + "_topicModel.lda")
      print "Time to run LDA with {0:d} topics is {1:.3f} seconds".format(numTopics, time.clock() - start)

      with open("../processed/" + parent[9:-1] + "_lda/" + directory + "_" + str(numTopics) + "_topics.txt", 'w') as outFile:
        topics = []
        for i in xrange(numTopics):
          topics.append(sorted(lda.show_topic(i, 20), reverse = True))
          
        for topic in topics:
          for word in topic:
            print >> outFile, word[1] + ", ", 
          print >> outFile, "\n"
      print "\n"      

def showHDPTopics(hdpmodel):
  topics = hdpmodel.show_topics(topics=-1, topn=20, formatted = False)
  for topic in topics:
    print "Topic", topic[0], ": ", 
    for wordTuple in topic[1]:
      print "{0:s} : {1:.4f}".format(wordTuple[0], wordTuple[1]),
    print "\n\n" 

def getCategories():
  dirnames = [["health", "eventservices", "pets" ,"publicservicesgovt" ,"nightlife" ,"shopping"]]
  dirnames.append(["homeservices", "education", "localflavor", "arts", "beautysvc", "financialservices", "localservices"])
  dirnames.append(["food", "auto", "active", "hotelstravel", "religiousorgs", "massmedia", "professional"])
  dirnames.append(["restaurants"])
  dirnames = [item for sublist in dirnames for item in sublist]
  return dirnames

def doAllHDP(parent):
  categories = getCategories()
  for category in categories:
    doHDP(parent, category)
    
def doHDP(parent, directory):
  # requires that dictionary and corpus file already exist
  files = os.listdir(parent + directory)
  dictionary = corpora.Dictionary.load("../processed/" + parent[9:-1] + "_lda/" + directory + ".dict")
  review_corpus = corpora.MmCorpus("../processed/" + parent[9:-1] + "_lda/" + directory + ".mm")

  numTopics = [3, 5, 10]
  for numTopic in numTopics:
    print "Running HDP for", directory, "for", numTopic, "topics\n"
    hdp = HdpModel(corpus=review_corpus, id2word=dictionary, T=numTopic, K=10, gamma = 0.8, alpha = 1)
    hdp.save("../processed/" + parent[9:-1] + "_hdp/" + directory + "_" + str(numTopic) + "_topicModel.hdp")
    showHDPTopics(hdp)  


def doLDA(parent, directory):
  
#  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  files = os.listdir(parent + directory)
  if len(files) > 1000:
    numTopics = 5
  else:
    numTopics = 3

  passes = 1
  if len(sys.argv) > 2 and sys.argv[2]:
    passes = int(sys.argv[2])
      
  dictionary = createDict(parent + directory + "/", files)
  
  createCorpus(parent, directory, dictionary)
  '''
  # NOTE: Multicore seemed to actually take longer to run than regular
  # 2735 seconds for multicore vs 1755 secs for regular on the low-rated restaurant review
  start = time.clock()  
  lda = models.ldamulticore.LdaMulticore(corpus=review_corpus, id2word=dictionary, 
                                 num_topics=numTopics, chunksize=10)  

  print "Execution time for multicore is ", time.clock() - start, "seconds"
  '''
  start = time.clock()
  if passes == 1:
    lda = models.ldamodel.LdaModel(corpus=review_corpus, id2word=dictionary, 
                                 num_topics=numTopics, chunksize=10)
  else: # batch LDA
    lda = models.ldamodel.LdaModel(corpus=review_corpus, id2word=dictionary, 
                                 num_topics=numTopics, chunksize=10, update_every=0, passes=passes)  
  print "Time to run LDA is {0:.3f} seconds".format(time.clock() - start)

  topics = []
  for i in xrange(numTopics):
    topics.append(sorted(lda.show_topic(i, 20), reverse = True))
    
  for topic in topics:
    for word in topic:
      print word[1] + ", ", 
    print "\n"
          
if __name__ == '__main__':
  print "Usage: python src/runGensim.py <topicName> <numPasses> <hdp>"
  print "Eg. python src/runGensim.py localflavor 1 hdp"
  
  if sys.argv[1].lower() == "--doall":
    if sys.argv[2].lower() == "lda":
      doAll("../train/low_reviews/")
      doAll("../train/high_reviews/")
    else:
#      doAllHDP("../train/low_reviews/")
      doAllHDP("../train/high_reviews/")
      
  else:  
    directory = sys.argv[1]

    if len(sys.argv) > 3 and sys.argv[3].lower() == "hdp":
      parent = "../train/low_reviews/"
      doHDP(parent, directory)
      parent = "../train/high_reviews/"
      doHDP(parent, directory)      
    else:
      parent = "../train/low_reviews/"
      doLDA(parent, directory)
      parent = "../train/high_reviews/"
      doLDA(parent, directory)