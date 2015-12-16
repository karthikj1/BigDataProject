'''
Created on Nov 19, 2015

@author: karthik
'''

import yaml
import sys

if __name__ == '__main__':  
  outDir = sys.argv[2] + "/"
  
  i = 1
  with open(sys.argv[1], 'r') as reviewFile:
    for line in reviewFile:            
      review = yaml.safe_load(line)['text']      
      with open(outDir + str(i) + ".txt", "w") as out:
        print >> out, review.lower().encode('UTF-8')
        out.flush()
      print "Wrote review", i, "\r", 
      i += 1
      