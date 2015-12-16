'''
Created on Nov 22, 2015

@author: karthik
Move reviews into their toplevel category
'''
import yaml
import json
import os

if __name__ == '__main__':
  f = open("yelp/yelp_categories.json", "r")
  cats = yaml.safe_load(f)
  toplevel = []
  for cat in cats:
   if len(cat['parents']) == 0:
     toplevel.append(cat['alias'])

  toplevelNames = []
  for cat in cats:
   if len(cat['parents']) == 0:
      toplevelNames.append(cat['title'])

  f = open('toplevel_categories.txt', 'w')
  print >> f, toplevel
  f.close()

  for cat in toplevel:
    os.mkdir("../high_reviews/" + cat)

  for cat in toplevel:
    os.mkdir("../low_reviews/" + cat)
  
  catDict = {toplevelNames[i]:toplevel[i] for i in xrange(len(toplevel))}
 
  # copy low rating reviews to their correct top-level category directory
  reviewCats = open('low_categories.txt', 'r')
  i = 0
  for line in reviewCats:
   cats = eval(line)
   i += 1
   if len(cats) == 0:
     continue
   targetDir = ""
   for cat in cats:
    if cat in catDict:
      targetDir = catDict[cat]
   srcFile = "../low_reviews/" + str(i) + ".txt"
   tgtFile = "../low_reviews/" + targetDir + "/" + str(i) + ".txt"
   if targetDir == "":
    continue
   os.rename(srcFile, tgtFile)

  # copy high rating reviews to their correct top-level category directory
  reviewCats = open('high_categories.txt', 'r')
 
  i = 0
  for line in reviewCats:
    cats = eval(line)
    i += 1
    if len(cats) == 0:
      continue
    targetDir = ""
    for cat in cats:
      if cat in catDict:
        targetDir = catDict[cat]
    srcFile = "../high_reviews/" + str(i) + ".txt"
    tgtFile = "../high_reviews/" + targetDir + "/" + str(i) + ".txt"
    if targetDir == "":
      continue
    os.rename(srcFile, tgtFile)   
