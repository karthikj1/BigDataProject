"""
pyLDAvis Gensim
===============
Helper functions to visualize LDA models trained by Gensim
"""

from __future__ import absolute_import
import funcy as fp
import numpy as np
import pandas as pd
from past.builtins import xrange
from pyLDAvis import prepare as vis_prepare

import scipy.spatial.distance as dist
from scipy.stats import entropy
from sklearn import manifold
from sklearn.metrics import pairwise


def _extract_data(topic_model, corpus, dictionary, doc_topic_dists=None):
   import gensim
   
   if not gensim.matutils.ismatrix(corpus):
      corpus_csc = gensim.matutils.corpus2csc(corpus)
   else:
      corpus_csc = corpus
   
   vocab = list(dictionary.token2id.keys())
   # TODO: add the hyperparam to smooth it out? no beta in online LDA impl.. hmm..
   # for now, I'll just make sure we don't ever get zeros...
   beta = 0.01
   fnames_argsort = np.asarray(list(dictionary.token2id.values()), dtype=np.int_)
   term_freqs = corpus_csc.sum(axis=1).A.ravel()[fnames_argsort]
   term_freqs[term_freqs == 0] = beta
   doc_lengths = corpus_csc.sum(axis=0).A.ravel()

   assert term_freqs.shape[0] == len(dictionary), 'Term frequencies and dictionary have different shape {} != {}'.format(term_freqs.shape[0], len(dictionary))
   assert doc_lengths.shape[0] == len(corpus), 'Document lengths and corpus have different sizes {} != {}'.format(doc_lengths.shape[0], len(corpus))

   if doc_topic_dists is None:
      gamma = topic_model.inference(corpus)
      doc_topic_dists = gamma / gamma.sum(axis=1)[:, None]

   # hack to eliminate NaN's
   dummy = np.asarray([1.0/doc_topic_dists.shape[1]] * doc_topic_dists.shape[1], np.float64)      
   for i in xrange(len(doc_topic_dists)):
     if np.isnan(doc_topic_dists[i]).any():
        doc_topic_dists[i] = dummy

   assert np.isnan(doc_topic_dists).any() == False, 'There were nan values in the doc_topic distributions'
   assert doc_topic_dists.shape[1] == len(topic_model.m_lambda), 'Document topics and number of topics do not match {} != {}'.format(doc_topic_dists.shape[0], topic_model.num_topics)

   # get the topic-term distribution straight from gensim without
   # iterating over tuples
   topic = topic_model.m_lambda
   topic = topic / topic.sum(axis=1)[:, None]
   topic_term_dists = topic[:, fnames_argsort]

   assert topic_term_dists.shape[0] == doc_topic_dists.shape[1]

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

def prepare(topic_model, corpus, dictionary, doc_topic_dist=None, **kwargs):
    """Transforms the Gensim TopicModel and related corpus and dictionary into
    the data structures needed for the visualization.

    Parameters
    ----------
    topic_model : gensim.models.hdpmodel.HdpModel
        An already trained Gensim HdpModel. 

    corpus : array-like list of bag of word docs in tuple form or scipy CSC matrix
        The corpus in bag of word form, the same docs used to train the model.
        The corpus is transformed into a csc matrix internally, if you intend to
        call prepare multiple times it is a good idea to first call
        `gensim.matutils.corpus2csc(corpus)` and pass in the csc matrix instead.

    For example: [(50, 3), (63, 5), ....]

    dictionary: gensim.corpora.Dictionary
        The dictionary object used to create the corpus. Needed to extract the
        actual terms (not ids).

    doc_topic_dist (optional): Document topic distribution from LDA (default=None)
        The document topic distribution that is eventually visualised, if you will
        be calling `prepare` multiple times it's a good idea to explicitly pass in
        `doc_topic_dist` as inferring this for large corpora can be quite
        expensive.

    **kwargs :
        additional keyword arguments are passed through to :func:`pyldavis.prepare`.

    Returns
    -------
    prepared_data : PreparedData
        the data structures used in the visualization

    Example
    --------
    For example usage please see this notebook:
    http://nbviewer.ipython.org/github/bmabey/pyLDAvis/blob/master/notebooks/Gensim%20Newsgroup.ipynb

    See
    ------
    See `pyLDAvis.prepare` for **kwargs.
    """
    # we use sklearn's multi-dimensional scaling as the default measure to approximate distance between topics
    # should be a slightly more stable implementation compared to skbio's PCoA 
    if 'mds' not in kwargs:
      kwargs['mds'] = js_MDS 
      
    opts = fp.merge(_extract_data(topic_model, corpus, dictionary, doc_topic_dist), kwargs)
    return vis_prepare(**opts)
