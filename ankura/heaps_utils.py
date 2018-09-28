import ankura
import string
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
REDDIT = 'reddit'
CONTROVERSIAL = 'controversiality'
AM_LRG = 'amazon_large'
AM_LRG_SAMPLE = f'{AM_LRG}_sample'
RATING = 'rating'
YELP = 'yelp'
NEWS = 'na_news'
NEWS_SAMPLE = f'{NEWS}_sample'
NAIVE_BAYES = 'nb'
SVM = 'svm'
AM_LRG_DIR = os.path.join(os.getenv('HOME'), 'compute/amazon_large')
NEWS_DIR = os.path.join(os.getenv('HOME'), 'compute/.ankura/na_news')
REDDIT_DIR = os.path.join(os.getenv('HOME'), 'compute/.ankura/reddit')
ANK_CORPUS_DICT = ankura.corpus.corpus_dictionary

DIR_CORPUS_DICT = {
                        NEWS : NEWS_DIR,
                        AM_LRG : AM_LRG_DIR,
                        REDDIT : REDDIT_DIR,
                        AM_LRG_SAMPLE : AM_LRG_DIR,
                        NEWS_SAMPLE : NEWS_DIR,
                  }

EXT_DICT = { 
                NEWS : 'txt',
                AM_LRG : 'json',
                AM_LRG_SAMPLE : 'json',
                NEWS_SAMPLE : 'txt',
           }

SK_MODEL_DICT =  {
                    NAIVE_BAYES : MultinomialNB,
                    SVM : SVC,
                 }

SK_LEARN_MODELS = {'nb', 'svm'}

LABEL_DICT = defaultdict(lambda: 'label')
LABEL_DICT[YELP] = RATING
LABEL_DICT[REDDIT] = CONTROVERSIAL

ExperimentResults = namedtuple('ExperimentResults', 'corpus model seed hash rare train test accuracy vocabulary all_time train_time test_time vocab_time')
PlottingData = namedtuple('PlottingData', 'types tokens std')

def boolparse(b):
    if b.lower() == 'true':
        return True
    elif b.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_sysargs():
    parser = argparse.ArgumentParser(description='Process arguments for type token calculation the user wants to run')
    parser.add_argument('--corpus', required=True, type=str, help='corpus the calculation will use', choices=['amazon_large', 'amazon_large_sample', 'na_news', 'na_news_sample', 'reddit'])
    parser.add_argument('--iterations', type=int, default=5, help='Number of times to filter through corpus and take sample')
    parser.add_argument('--sample_size', type=int, help='Size of sample to take from corpus')
    parser.add_argument('--udreplace', type=boolparse, default=False, help='whether to replace \'_\' and \'-\' with \' \'')
    parser.add_argument('--nopunct', type=boolparse, default=False, help='whether to remove punctuation')
    parser.add_argument('--stop', type=boolparse, default=False, help='whether to remove stopwords')
    parser.add_argument('--stem', type=boolparse, default=False, help='whether to stem words')
    parser.add_argument('--lower', type=boolparse, default=False, help='whether to lowercase the tokens')
    parser.add_argument('--correct', type=boolparse, default=False, help='whether to correct the tokens')
    parser.add_argument('--nremoval', type=boolparse, default=False, help='whether to remove numbers')
    parser.add_argument('--segment', type=boolparse, default=False, help='whether to segment words')
    parser.add_argument('--model', type=str, help='ML model used by experiment', choices=['svm', 'nb', 'ankura'])
    parser.add_argument('--seed', type=int, default=0, nargs='?', help='seed for the random number generator')
    parser.add_argument('--hash', type=int, default=0, help='final hashed vocabulary size')
    parser.add_argument('--rare', type=int, default=0, help='rare word filter')
    parser.add_argument('--ntrain', type=int, help='number of documents in the training set')
    parser.add_argument('--ntest', type=int, help='number of documents in the test set')
    return parser.parse_args()


def sklearn_run(sysargs):
    return sysargs.model in SK_LEARN_MODELS


def get_corpus_file(sysargs, ext='txt'):
    base_name = os.path.join(DIR_CORPUS_DICT[sysargs.corpus], sysargs.corpus)
    if sysargs.correct: base_name = f'{base_name}_symspell'
    if sysargs.segment: base_name = f'{base_name}_wordseg'
    if sysargs.lower and not (sysargs.correct or sysargs.segment): base_name = f'{base_name}_lower'
    if sysargs.nopunct: base_name = f'{base_name}_nopunct'
    if sysargs.udreplace: base_name = f'{base_name}_udreplace'
    if sysargs.nremoval: base_name = f'{base_name}_nremoval'
    if sysargs.stop: base_name = f'{base_name}_stop'
    if sysargs.stem: base_name = f'{base_name}_stem'
    return f'{base_name}.{ext}.gz'
