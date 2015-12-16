import gensim
import pyLDAvis.gensim
import gensimHDP
import sys
import numpy as np
import mahoutLDA
import traceback
import os

def makeHTMLFromMahout(dir, category, numTopics):
  dictionary = gensim.corpora.Dictionary.load("mahout-" + dir + '/' + category + '.dict')
  corpus = gensim.corpora.MmCorpus("mahout-" + dir + '/' + category + '.mm')

  with open("mahout-" + dir + '/' + category + "_" + str(numTopics) + '_doctopic_dist.json', 'r') as dtFile:
    with open("mahout-" + dir + '/' + category + "_" + str(numTopics) + '_topicterm_dist.json', 'r') as topicTermFile:
      print dictionary
      preparedLDA = mahoutLDA.prepare(corpus, dictionary, dtFile, topicTermFile)
  
  pyLDAvis.save_html(preparedLDA, "mahout-" + dir + '/' + category + '_' + str(numTopics) + '_mahoutLDA.html')  

def makeHTML(dir, category, numTopics):
  dictionary = gensim.corpora.Dictionary.load(dir + '/' + category + '.dict')
  corpus = gensim.corpora.MmCorpus(dir + '/' + category + '.mm')
  lda = gensim.models.ldamodel.LdaModel.load(dir + '/' + category + '_' + str(numTopics) + '_topicModel.lda')

  preparedLDA = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
  # pyLDAvis.show(preparedLDA, n_retries = 1)
  pyLDAvis.save_html(preparedLDA, dir + '/' + category + '_' + str(numTopics) + '.html')  

def makeHTMLfromHDP(dir, category, numTopics,dictionary, corpus):
  print "using model", "hdp-" + dir + '/' + category + '_' + str(numTopics) + '_topicModel.hdp'
  hdp = gensim.models.hdpmodel.HdpModel.load("hdp-" + dir + '/' + category + '_' + str(numTopics) + '_topicModel.hdp')
  # use the saved doc_topic_dist matrix if it exists
  doc_topic_dist_path = "hdp-" + dir + '/' + category + '_' + str(numTopics) + '_dtd.npy'
  if os.path.exists(doc_topic_dist_path):
    with open(doc_topic_dist_path, 'r') as f:
      doc_topic_dist = np.load(f)
      preparedHDP = gensimHDP.prepare(hdp, corpus, dictionary, doc_topic_dist)
  else:
      preparedHDP = gensimHDP.prepare(hdp, corpus, dictionary)
  # pyLDAvis.show(preparedLDA, n_retries = 1)
  pyLDAvis.save_html(preparedHDP, "hdp-" + dir + '/' + category + '_' + str(numTopics) + '_hdp.html')  

def getCategories():
  dirnames = [["health", "eventservices", "pets" ,"publicservicesgovt" ,"nightlife" ,"shopping"]]
  dirnames.append(["homeservices", "education", "localflavor", "arts", "beautysvc", "financialservices", "localservices"])
  dirnames.append(["food", "auto", "active", "hotelstravel", "religiousorgs", "massmedia", "professional"])
  dirnames.append(["restaurants"])
  
  dirnames = [item for sublist in dirnames for item in sublist]
  
#  dirnames = ["localflavor"]
  return dirnames

if __name__ == '__main__':    
  dirnames = getCategories()
  
  doHDP = False
  if len(sys.argv) > 1 and sys.argv[1].lower() == "hdp":
    doHDP = True
  
  doMahout = False
  if len(sys.argv) > 1 and sys.argv[1].lower() == "mahout":
    doMahout = True
     
  doLDA = not(doMahout or doHDP)
  
  numTopicsChoices = [3, 5, 10]

  f = open('errors.txt', 'w')
  for category in dirnames:
      if doHDP:
        try:
          dictionary = gensim.corpora.Dictionary.load('low/' + category + '.dict')
          print "Loading corpus", 'low/' + category + '.mm'
          corpus = gensim.corpora.MmCorpus('low/' + category + '.mm')
          for numTopics in numTopicsChoices:
            print "Using HDP model on low", category, "with", numTopics
            makeHTMLfromHDP("low", category, numTopics, dictionary, corpus)
          
          dictionary = gensim.corpora.Dictionary.load("high/" + category + '.dict')
          print "Loading corpus", 'high/' + category + '.mm'
          corpus = gensim.corpora.MmCorpus("high" + '/' + category + '.mm')
          for numTopics in numTopicsChoices:
            print "Using HDP model on high", category, "with", numTopics
            makeHTMLfromHDP("high", category, numTopics, dictionary, corpus)
        except KeyboardInterrupt:
          f.close()
          quit()
        except:
          traceback.print_exc()
          print >> f, traceback.print_exc()
          print "Failed on", category, "with", numTopics
          print >> f, "Failed on", category, "with", numTopics
          f.flush()
          continue
      if doLDA:
        print "Running using Gensim LDA results"
        for numTopics in numTopicsChoices:
  #        makeHTML("low", category, numTopics)
          makeHTML("high", category, numTopics)
        
      if doMahout:
        print "Running", category, "using Mahout results"
        for numTopics in numTopicsChoices:
          try:
#            makeHTMLFromMahout("low", category, numTopics)
            makeHTMLFromMahout("high", category, numTopics)
          except KeyboardInterrupt:
            f.close()
            quit()
          except:
            traceback.print_exc()
            print >> f, traceback.print_exc()
            print "Failed on", category, "with", numTopics
            print >> f, "Failed on", category, "with", numTopics
            continue

