'''
Created on Nov 22, 2015

@author: karthik

'''

from subprocess import call
import sys
import operator

# export MAHOUT_HEAPSIZE=5000 to give Mahout more memory. Default is 4G

def createSequenceFiles(dirnames):
  parent = "file:///Courses/CSEE6893/Project/"
  inpParent = parent + "high_reviews/"
  outParent = parent + "high_reviews_seqFile/"
  for dirname in dirnames:
    inpDir = inpParent + dirname
    outDir = outParent + dirname
    print "\n\nCreating sequence file for", dirname, "\n"
    call(["bin/mahout", "seqdirectory", "-i", inpDir, "-o", outDir, "-c", "UTF-8", "-xm", "sequential", "-ow"])

def createTFVectors(dirnames, parent, MIN_DOCS, minSupport, maxDF):  
  # create TF vectors with terms appearing a minimum of minSupport times in a minimum of MIN_DOCS documents
  # and a max doc freq of maxDF
  # norm = "2"
  
  for dirname in dirnames:
    inpDir = parent + dirname
    outDir = parent + dirname + "/vectors_" + MIN_DOCS + "_" + minSupport + "_" + maxDF
    print "\n\nCreating vectors for", dirname, "\n"
    
    
    call(["bin/mahout", "seq2sparse", "-i", inpDir, "-o", outDir, "-wt", "tf", 
          "-s", minSupport, "-md", MIN_DOCS, "-x", maxDF, "-seq", "-ow"])

def createRowID(dirnames, parent, MIN_DOCS, minSupport, maxDF):  
  
  for dirname in dirnames:
    inpDir = parent + dirname + "/vectors_" + MIN_DOCS + "_" + minSupport + "_" + maxDF + "/tf-vectors"
    outDir = parent + dirname + "/matrix"
    print "\n\nConverting vectors for", dirname, "\n"    
    
    call(["bin/mahout", "rowid", "-i", inpDir, "-o", outDir])

def showTopics(termTopicFile):
  terms = {}
  f = open(termTopicFile, 'rb')
  ln = 0
  for line in f:
    if len(line.strip()) == 0: continue
    if ln == 0:
      # make {id,term} dictionary for use later
      tn = 0
      for term in line.strip().split(","):
        terms[tn] = term
        tn += 1
    else:
      # parse out topic and probability, then build map of term to score
      # finally sort by score and print top 10 terms for each topic.
      topic, probs = line.strip().split("\t")
      termProbs = {}
      pn = 0
      for prob in probs.split(","):
        termProbs[terms[pn]] = float(prob)
        pn += 1
      toptermProbs = sorted(termProbs.iteritems(),
        key=operator.itemgetter(1), reverse=True)
      print "Topic: %s" % (topic)
      print "\n".join([(" "*3 + x[0]) for x in toptermProbs[0:10]])
    ln += 1
  f.close()

if __name__ == '__main__':
  f = open('/cygdrive/c/Courses/CSEE6893/Project/toplevel_categories.txt', 'r')
  dirnames = eval(f.readline())
  f.close()

  maxDF = "75"
  MIN_DOCS = "5"
  minSupport = "10" 

  if sys.argv[1].lower() == "-seq":
    createSequenceFiles(dirnames)
    
  if sys.argv[1].lower() == "-vec":    
    createTFVectors(dirnames, sys.argv[2] + "/", MIN_DOCS, minSupport, maxDF)

  if sys.argv[1].lower() == "-row":    
    createRowID(dirnames,  "/low_reviews/", MIN_DOCS, minSupport, maxDF)
    createRowID(dirnames,  "/high_reviews/", MIN_DOCS, minSupport, maxDF)

  if sys.argv[1].lower() == "-topics":    
    showTopics('/cygdrive/c/ldatest')


  
