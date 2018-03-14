"""

Author: Sam
Date: Feb 2018

This file should be the only feel that users make any changes to.

Input data should be in a csv file with three columns: message id, message text, topics
"""

from datetime import datetime
import os

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

platform = "BOTH"  # Options are "TW", "FB", or "BOTH"
tw_input_data_file = os.path.join(os.path.dirname(__file__),'../../data/tw_fixed.csv')  # Should be a csv file
fb_input_data_file = os.path.join(os.path.dirname(__file__),'../../data/fb_fixed.csv')  # Should be a csv file

output_dir = '/Users/samjackson/Google Drive/Projects/Illuminating 2016/Topic/post_SMS/ML_from_scratch/' + str(now) + '/'

tw_feature_file = output_dir + 'tw_features.csv'  # Will be a csv file created by GetFeatures
fb_feature_file = output_dir + 'fb_features.csv'  # Will be a csv file created by GetFeatures
comb_feature_file = output_dir + 'comb_features.csv'  # Will be a csv file created by GetFeatures

# Pre-processing
stopwords_file = ''  # Should be a plain text file. If none provided, use NLTK English stopwords + "rt".

# Primary features
num_unigrams = 3000
num_bigrams = 2000
pos_restriction = False
pos_tags = ["NOUN", "ADJ"]

# Additional features
pos_counts = False
use_mentions = False
use_ne_chunks = True

''' 
Full List of POS tags: 
    VERB - verbs (all tenses and modes)
    NOUN - nouns (common and proper)
    PRON - pronouns
    ADJ - adjectives
    ADV - adverbs
    ADP - adpositions (prepositions and postpositions)
    CONJ - conjunctions
    DET - determiners
    NUM - cardinal numbers
    PRT - particles or other function words
    X - other: foreign words, typos, abbreviations
    . - punctuation
'''

# Model parameters
classifier = 'LinearSVC'
k_folds = 5

'''
Classifier options:
    SGD
    LinearSVC
    MNB
    BNB
    logit
'''

