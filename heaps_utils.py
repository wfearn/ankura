import ankura
from ankura.pipeline import HashedVocabBuilder
from ankura.pipeline import VocabBuilder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from collections import namedtuple
from collections import defaultdict
import os
import sys
import argparse

NUM_TOPICS = 80
LABEL = 'label'
RATING = 'rating'
AM_LRG = 'amazon_large'
YELP = 'yelp'
NEWS = 'na_news'
NAIVE_BAYES = 'nb'
SVM = 'svm'
AM_LRG_C = f'{AM_LRG}_corrected'
AM_LRG_TEST = f'{AM_LRG}_test'
AM_LRG_SAMPLE = f'{AM_LRG}_sample'
AM_LRG_SYM_NP = f'{AM_LRG}_symspell_nopunct'
NEWS_C = f'{NEWS}_corrected'
HOME_DIR = os.environ['HOME']
AM_LRG_DIR = os.path.join(HOME_DIR, f'compute/{AM_LRG}')
NEWS_DIR = os.path.join(HOME_DIR, f'compute/.ankura/{NEWS}')
AM_LRG_FILE = f'{AM_LRG}_text.json'
AM_LRG_TEST_FILE = f'{AM_LRG_TEST}.json'
AM_LRG_SAMPLE_FILE = f'{AM_LRG_SAMPLE}.json'
AM_LRG_SYM_NP_FILE = f'{AM_LRG_SYM_NP}.txt'
NEWS_FILE = f'{NEWS}.txt'
AM_LRG_CORRECTED_FILE = f'{AM_LRG_C}.json'
NEWS_CORRECTED_FILE = f'{NEWS_C}.txt'
AM_LRG_FILEPATH = os.path.join(AM_LRG_DIR, AM_LRG_FILE)
AM_LRG_CORRECTED_FILEPATH = os.path.join(AM_LRG_DIR, AM_LRG_CORRECTED_FILE)
AM_LRG_TEST_FILEPATH = os.path.join(AM_LRG_DIR, AM_LRG_TEST_FILE)
AM_LRG_SAMPLE_FILEPATH = os.path.join(AM_LRG_DIR, AM_LRG_SAMPLE_FILE)
AM_LRG_SYM_NP_FILEPATH = os.path.join(AM_LRG_DIR, AM_LRG_SYM_NP_FILE)
NEWS_FILEPATH = os.path.join(NEWS_DIR, NEWS_FILE)
NEWS_CORRECTED_FILEPATH = os.path.join(NEWS_DIR, NEWS_CORRECTED_FILE)
ANK_CORPUS_DICT = ankura.corpus.corpus_dictionary

FILE_CORPUS_DICT = {
                        AM_LRG : AM_LRG_FILEPATH,
                        NEWS : NEWS_FILEPATH, 
                        NEWS_C : NEWS_CORRECTED_FILEPATH,
                        AM_LRG_C : AM_LRG_CORRECTED_FILEPATH,
                        AM_LRG_TEST : AM_LRG_TEST_FILEPATH,
                        AM_LRG_SAMPLE : AM_LRG_SAMPLE_FILEPATH,
                        AM_LRG_SYM_NP : AM_LRG_SYM_NP_FILEPATH,
                   }

SK_MODEL_DICT =  {
                    NAIVE_BAYES : MultinomialNB,
                    SVM : SVC,
                 }

SK_LEARN_MODELS = {'nb', 'svm'}

LABEL_DICT = defaultdict(lambda x: 'label')
LABEL_DICT[YELP] = RATING

ExperimentResults = namedtuple('ExperimentResults', 'corpus model seed hash rare train test accuracy vocabulary all_time train_time test_time vocab_time')
PlottingData = namedtuple('PlottingData', 'types tokens std')

def create_tokenizer(sysargs):
    t = ankura.pipeline.split_tokenizer()
    if sysargs.udreplace: tokenizer = ankura.pipeline.sub_tokenizer(t, '[_|-]+', ' ')
    if sysargs.nremoval: tokenizer = ankura.pipeline.sub_tokenizer(t, '[\d]+', '')
    t = ankura.pipeline.translate_tokenizer(t)

    return t
