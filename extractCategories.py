'''
Created on Nov 19, 2015

@author: karthik
'''

import yaml
import sys

if __name__ == '__main__':  
  outFile = "low_cats.txt"
  
  i = 1
  with open(sys.argv[1], 'r') as reviewFile:
    with open(outFile, "w") as out:
      for line in reviewFile:            
        categories = yaml.safe_load(line)['categories']      
        print >> out, categories
        out.flush()
        print "Wrote review", i, "\r", 
        i += 1

  print "\n"
