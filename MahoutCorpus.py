'''
Created on Nov 25, 2015

@author: karthik
'''
import os
import re

class MahoutCorpus(object):

  def __init__(self, directory, nonWordRegex, dictionary):
    self.directory = directory
    # list of files that went missing in Mahout's seq2sparse (probably because they are empty)- we ignore these files
    self.low_missing = {'arts': {'246038.txt': True, '109739.txt': True}, 'shopping': {'287734.txt': True, '98067.txt': True, '216446.txt': True, '48091.txt': True}, 'homeservices': {'259135.txt': True, '41557.txt': True, '123296.txt': True}, 'beautysvc': {'20202.txt': True, '202722.txt': True, '275490.txt': True, '289592.txt': True}, 'financialservices': {}, 'auto': {'26160.txt': True, '201410.txt': True, '126229.txt': True, '281976.txt': True}, 'restaurants': {}, 'religiousorgs': {}, 'localservices': {}, 'food': {'99682.txt': True, '188381.txt': True}, 'professional': {}, 'health': {'203235.txt': True, '202626.txt': True}, 'eventservices': {'79801.txt': True, '21245.txt': True, '166118.txt': True, '211963.txt': True, '60222.txt': True, '104845.txt': True}, 'pets': {'23640.txt': True}, 'nightlife': {'263063.txt': True, '2085.txt': True, '284508.txt': True, '5988.txt': True, '89085.txt': True, '253956.txt': True, '256441.txt': True}, 'active': {}, 'hotelstravel': {'146566.txt': True, '63548.txt': True, '260333.txt': True}, 'education': {}, 'publicservicesgovt': {}, 'localflavor': {}, 'massmedia': {}}
    self.high_missing = {'arts': {'962107.txt': True}, 'shopping': {'38174.txt': True, '890344.txt': True}, 'homeservices': {'371428.txt': True, '385344.txt': True}, 'beautysvc': {'676805.txt': True, '734848.txt': True, '833676.txt': True}, 'financialservices': {}, 'auto': {'170887.txt': True, '749425.txt': True, '941648.txt': True, '382282.txt': True}, 'restaurants': {}, 'religiousorgs': {}, 'localservices': {'387207.txt': True, '989059.txt': True, '439389.txt': True}, 'food': {'821347.txt': True, '543035.txt': True, '912247.txt': True, '646806.txt': True, '524675.txt': True, '49930.txt': True, '602448.txt': True, '302462.txt': True, '912246.txt': True, '978604.txt': True, '725313.txt': True, '485151.txt': True}, 'professional': {}, 'health': {'863451.txt': True, '53367.txt': True}, 'eventservices': {'73772.txt': True, '407425.txt': True, '90783.txt': True, '243618.txt': True}, 'pets': {'35238.txt': True}, 'nightlife': {'746369.txt': True, '975272.txt': True, '920019.txt': True, '709974.txt': True, '135550.txt': True}, 'active': {'103820.txt': True, '605487.txt': True, '1015154.txt': True, '275974.txt': True}, 'hotelstravel': {'573405.txt': True}, 'education': {}, 'publicservicesgovt': {'320657.txt': True, '578433.txt': True}, 'localflavor': {}, 'massmedia': {}}
    
    self.files = os.listdir(directory)
    # extract dirname eg. arts from directory path fs/sdf/auto/
    dirname = directory[directory[:-1].rfind('/') + 1: -1]
    self.files = [fn for fn in self.files 
                  if fn not in self.low_missing[dirname] and fn not in self.high_missing[dirname]]
    self.dictionary = dictionary
    self.nonWordRegex = nonWordRegex

      
  def __iter__(self):     
    for filename in self.files:
      if not os.path.isfile(self.directory + filename): 
          continue         
      for line in open(self.directory + filename):
        # assume there's one document per line, tokens separated by nonWordRegex
        yield self.dictionary.doc2bow(self.nonWordRegex.split(line))  
  
  def __len__(self):
    return len(self.files)