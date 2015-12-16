'''
Created on Nov 25, 2015

@author: karthik
'''
import os
import re

class MyCorpus(object):

  def __init__(self, directory, nonWordRegex, dictionary):
    self.directory = directory
    self.files = os.listdir(directory)
    self.nonWordRegex = nonWordRegex
    self.dictionary = dictionary
      
  def __iter__(self):     
    for filename in self.files:
      if not os.path.isfile(self.directory + filename): 
          continue               
      for line in open(self.directory + filename):
        # assume there's one document per line, tokens separated by whitespace
        yield self.dictionary.doc2bow(self.nonWordRegex.split(line))  
  
  def __len__(self):
    return len(self.files)