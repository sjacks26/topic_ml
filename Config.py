"""

Author: Sam
Date: Nov 2017
This file sets parameters and locates input files (including data and stopwords)

"""

from datetime import datetime

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

platform = "BOTH" # Options are "TW", "FB", or "BOTH"
tw_input_data_file = '/Users/samjackson/Google Drive/Projects/Illuminating 2016/Topic/post_SMS/data/tw_fixed.csv' # Should be a csv file
fb_input_data_file = '/Users/samjackson/Google Drive/Projects/Illuminating 2016/Topic/post_SMS/data/fb_fixed.csv' # Should be a csv file
stopwords_file = '' # Should be a plain text file. If none provided, use NLTK English stopwords.
tw_feature_file = '/Users/samjackson/Google Drive/Projects/Illuminating 2016/Topic/post_SMS/data/tw_for_ml_features' + str(now) + '.csv' # Should be a csv file created by GetFeatures
fb_feature_file = '/Users/samjackson/Google Drive/Projects/Illuminating 2016/Topic/post_SMS/data/fb_for_ml_features' + str(now) + '.csv' # Should be a csv file created by GetFeatures
comb_feature_file = '/Users/samjackson/Google Drive/Projects/Illuminating 2016/Topic/post_SMS/data/comb_ml_features' + str(now) + '.csv' # Should be a csv file created by GetFeatures

# Primary features
num_unigrams = 2000
num_bigrams = 1000
pos_restriction = False
pos_tags = ["NOUN", "ADJ"]

# Additional features
pos_counts = True
use_mentions = False

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

k_folds = 5

classifier = 'LinearSVC'

'''
Classifier options:
    SGD
    LinearSVC
    MNB
    BNB
    logit
'''

