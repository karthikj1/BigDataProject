"""
pyLDAvis Mahout LDA
===============
Helper functions to visualize LDA models trained by Mahout
"""

from __future__ import absolute_import
import funcy as fp
import numpy as np
import pandas as pd
from past.builtins import xrange
from pyLDAvis import prepare as vis_prepare

import gensim
import scipy.spatial.distance as dist
from scipy.stats import entropy
from sklearn import manifold
from sklearn.metrics import pairwise
from operator import itemgetter

def get_doc_topic(dtFile):
  # skip first two lines - they are just headers from Mahout
  dtFile.readline()
  dtFile.readline()
  
  mat = []
  for line in dtFile:
    line = line[line.find('{') + 1:-2]
    row = [float(v) for k, v in (x.split(':') for x in line.split(','))]
    mat.append(row)
  
  outArray = np.array(mat[:-1], dtype=np.float64) #Last row is count so ignore that
  return outArray

def get_topic_term(topicTermfile, dictionary):
    
  vocabSize = len(dictionary.token2id.keys())
  mat = []
  for line in topicTermfile:
    line = line.strip()[1:-1] # strip out leading and trailing { }
    vals = [x.split(':') for x in line.split(",")]
    # if mahout has a term that includes a comma, the splitting gets messed up
    # so we just throw out those terms - this is consistent with what the dictionary contains
    vals = [pair for pair in vals if len(pair) == 2] 
    # need to convert to unicode to capture foreign characters
    row = [(dictionary.token2id[unicode(kvpair[0], 'utf-8')], float(kvpair[1])) for kvpair in vals]
    topicRow = [None] * vocabSize
    for item in row:
      topicRow[item[0]] = item[1]
    mat.append(topicRow)
    
  outArray = np.array(mat, dtype = np.float64)
    
  return outArray

def _extract_data(corpus, dictionary, doc_topic_dists, topic_term_dists):
   
   if not gensim.matutils.ismatrix(corpus):
      corpus_csc = gensim.matutils.corpus2csc(corpus)
   else:
      corpus_csc = corpus
   
   vocab = list(dictionary.token2id.keys())

   beta = 0.01
   fnames_argsort = np.asarray(dictionary.token2id.values(), dtype=np.int_)
   term_freqs = corpus_csc.sum(axis=1).A.ravel()[fnames_argsort]
   term_freqs[term_freqs == 0] = beta    # for now, I'll just make sure we don't ever get zeros...
   doc_lengths = corpus_csc.sum(axis=0).A.ravel()

   assert term_freqs.shape[0] == len(vocab), 'Term frequencies and dictionary have different shape {} != {}'.format(term_freqs.shape[0], len(dictionary))
   assert doc_lengths.shape[0] == len(corpus), 'Document lengths and corpus have different sizes {} != {}'.format(doc_lengths.shape[0], len(corpus))

   print "Doc length is", len(doc_lengths), "or", doc_lengths.shape
   print "dtd length is", doc_topic_dists.shape
#   assert topic_term_dists.shape[0] == doc_topic_dists.shape[1], 'Document topics and number of topics in topic terms do not match {} != {}'.format(topic_term_dists.shape[0], doc_topic_dists.shape[1])

   return {'topic_term_dists': topic_term_dists, 'doc_topic_dists': doc_topic_dists,
           'doc_lengths': doc_lengths, 'vocab': vocab, 'term_frequency': term_freqs}

def _jensen_shannon(_P, _Q):
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def js_MDS(distributions):
   """Dimension reduction via Jensen-Shannon Divergence & multi-dimensional scaling

    Parameters
    ----------
    distributions : array-like, shape (`n_dists`, `k`)
        Matrix of distributions probabilities.

    Returns
    -------
    mds : array, shape (`n_dists`, 2)
   """
   dm = pairwise.pairwise_distances(distributions.values, metric = _jensen_shannon)
   dm[np.isinf(dm)] = 0 # bit questionable if this is the right way to handle infinite distances 
   mdscalc = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed")
   pos = mdscalc.fit_transform(dm)
   
   return pos

def prepare(corpus, dictionary, doc_topic_file, topic_term_file, **kwargs):
    """Transforms the Mahout LDA and related corpus and dictionary into
    the data structures needed for the visualization.

    Parameters
    ----------

    corpus : array-like list of bag of word docs in tuple form or scipy CSC matrix
        The corpus in bag of word form, the same docs used to train the model.
        The corpus is transformed into a csc matrix internally, if you intend to
        call prepare multiple times it is a good idea to first call
        `gensim.matutils.corpus2csc(corpus)` and pass in the csc matrix instead.

    For example: [(50, 3), (63, 5), ....]

    dictionary: gensim.corpora.Dictionary
        The dictionary object used to create the corpus. Needed to extract the
        actual terms (not ids).
    
    doc_topic_file : file handle to Document topic distribution from Mahout LDA 
        The document topic distribution that is eventually visualised

    topic_term_file : file handle to topic term distribution from Mahout LDA 
        The document topic distribution that is eventually visualised

    **kwargs :
        additional keyword arguments are passed through to :func:`pyldavis.prepare`.

    Returns
    -------
    prepared_data : PreparedData
        the data structures used in the visualization

    See
    ------
    See `pyLDAvis.prepare` for **kwargs.
    """
    # we use sklearn's multi-dimensional scaling as the default measure to approximate distance between topics
    # should be a slightly more stable implementation compared to skbio's PCoA 
    if 'mds' not in kwargs:
      kwargs['mds'] = js_MDS 
    
    doc_topic_dist = get_doc_topic(doc_topic_file)
    topic_term_dists = get_topic_term(topic_term_file, dictionary)
    
    opts = fp.merge(_extract_data(corpus, dictionary, doc_topic_dist, topic_term_dists), kwargs)
    return vis_prepare(**opts)
