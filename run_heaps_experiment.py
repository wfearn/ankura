import ankura
from ankura.pipeline import HashedVocabBuilder
from ankura.pipeline import VocabBuilder
import json
import numpy as np
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import namedtuple
import time
import os
import sys
import pickle
import argparse


AM_LRG = 'amazon_large'
NEWS = 'na_news'
NAIVE_BAYES = 'nb'
SVM = 'svm'
AM_LRG_C = f'{AM_LRG}_corrected'
AM_LRG_TEST = f'{AM_LRG}_test'
NEWS_C = f'{NEWS}_corrected'
HOME_DIR = os.environ['HOME']
AM_LRG_DIR = os.path.join(HOME_DIR, f'compute/{AM_LRG}')
NEWS_DIR = os.path.join(HOME_DIR, f'compute/.ankura/{NEWS}')
AM_LRG_FILE = f'{AM_LRG}_text.json'
AM_LRG_TEST_FILE = f'{AM_LRG_TEST}.json'
NEWS_FILE = f'{NEWS}.txt'
AM_LRG_CORRECTED_FILE = f'{AM_LRG_C}.json'
NEWS_CORRECTED_FILE = f'{NEWS_C}.txt'
AM_LRG_FILEPATH = os.path.join(AM_LRG_DIR, AM_LRG_FILE)
AM_LRG_CORRECTED_FILEPATH = os.path.join(AM_LRG_DIR, AM_LRG_CORRECTED_FILE)
AM_LRG_TEST_FILEPATH = os.path.join(AM_LRG_DIR, AM_LRG_TEST_FILE)
NEWS_FILEPATH = os.path.join(NEWS_DIR, NEWS_FILE)
NEWS_CORRECTED_FILEPATH = os.path.join(NEWS_DIR, NEWS_CORRECTED_FILE)
ANK_CORPUS_DICT = ankura.corpus.corpus_dictionary

FILE_CORPUS_DICT = {
                        AM_LRG : AM_LRG_FILEPATH,
                        NEWS : NEWS_FILEPATH, 
                        NEWS_C : NEWS_CORRECTED_FILEPATH,
                        AM_LRG_C : AM_LRG_CORRECTED_FILEPATH,
                        AM_LRG_TEST : AM_LRG_TEST_FILEPATH,
                   }

SK_MODEL_DICT =  {
                    NAIVE_BAYES : MultinomialNB,
                    SVM : SVC,
                 }

SK_LEARN_MODELS = {'nb', 'svm'}


ExperimentResults = namedtuple('ExperimentResults', 'corpus model seed hash rare train test accuracy all_time train_time test_time vocab_time')

def get_sysargs():
    parser = argparse.ArgumentParser(description='Process arguments for the experiment the user wants to run.')
    parser.add_argument('--corpus', type=str, required=True, help='the corpus the experiment will use', choices=['amazon_large', 'na_news', 'twitter', 'amazon_large_corrected', 'amazon_large_test'])
    parser.add_argument('--model', type=str, required=True, help='ML model used by experiment', choices=['svm', 'nb', 'ankura'])
    parser.add_argument('--seed', type=int, default=0, nargs='?', help='seed for the random number generator')
    parser.add_argument('--hash', type=int, default=0, help='final hashed vocabulary size')
    parser.add_argument('--rare', type=int, default=0, help='rare word filter')
    parser.add_argument('--udreplace', help='whether to replace \'_\' and \'-\' with \' \'')
    parser.add_argument('--nremoval', help='whether to remove numbers')
    parser.add_argument('--ntrain', type=int, help='number of documents in the training set')
    parser.add_argument('--ntest', type=int, help='number of documents in the test set')

    return parser.parse_args()


def format_path(cmd_args, filename):
    return os.path.join(HOME_DIR, f'.ankura/{cmd_args.corpus}/{cmd_args.model}/{filename}')


def get_sklearn_corpus(corpus_name):
    return open(FILE_CORPUS_DICT[corpus_name], 'r')


def get_ankura_corpus(args):
    return ANK_CORPUS_DICT[args.corpus](args)


def get_json_values(json_string, text_tag='reviewText', score_tag='overall'):
    s = json.loads(json_string)
    return s[text_tag], s[score_tag]


def get_text_values(text):
    return text, np.random.randint(6)


def get_doc_target(doc_list, is_text, vbuilder, tokenizer, end, start=None):
    subset = doc_list[:end] if not start else doc_list[start:end]
    vocabulary = vbuilder.tokens
    
    text_list = list()
    target_list = list()
    for doc in subset:
        text, score = get_text_values(doc) if is_text else get_json_values(doc)
        text_list.append(' '.join([vocabulary[t.token] for t in vbuilder.convert(tokenizer(text))]))
        target_list.append(score)

    return text_list, target_list


def run_experiment(args):
    overall_time_start = time.time()
    print('Running experiment')
    if args.model in SK_LEARN_MODELS: 
        corpus = get_sklearn_corpus(args.corpus)

        tokenizer = ankura.pipeline.split_tokenizer()
        if args.udreplace: tokenizer = ankura.pipeline.regex_tokenizer(tokenizer, '[_|-]+', ' ')
        if args.nremoval: tokenizer = ankura.pipeline.regex_tokenizer(tokenizer, '[\d]+', '')
        tokenizer = ankura.pipeline.translate_tokenizer(tokenizer)

        vb = HashedVocabBuilder(args.hash) if args.hash else VocabBuilder()

        print('Setting up document list')
        doc_list = list()
        for doc in corpus: #assumes each line in corpus file is a a document
            doc_list.append(doc)
        np.random.shuffle(doc_list)

        is_text = True if FILE_CORPUS_DICT[args.corpus].endswith('.txt') else False

        print('Getting train test split')
        train, train_target = get_doc_target(doc_list, is_text, vb, tokenizer, args.ntrain)
        test, test_target = get_doc_target(doc_list, is_text, vb, tokenizer, args.ntrain + args.ntest, start=args.ntrain)

        v = TfidfVectorizer()

        print('Preprocessing text')
        vocab_start = time.time()
        v.fit(train + test)
        train_vector = v.transform(train)
        test_vector = v.transform(test)
        vocab_end = time.time()

        model = SK_MODEL_DICT[args.model]()

        print('Fitting model')
        train_start = time.time()
        model.fit(train_vector, train_target)
        train_end = time.time()

        test_start = time.time()
        accuracy = model.score(test_vector, test_target)
        test_end = time.time()
        print('Accuracy is:', accuracy)


    else:
        corpus = get_ankura_corpus(args)
        print('Splitting corpus')
        split, test = ankura.pipeline.train_test_split(corpus, random_seed=args.seed, num_train=args.ntrain, num_test=args.ntest, vocab_size=args.hash)
    

    """   Process for ankura
          need a way to run tokenizer based on parameters from args
            load corpus
            build Q
            get topics
            topic assignments
            build doc/topic matrix
            load doc/topic into LogisticRegression
            score model
    """

    overall_time_end = time.time()

    all_time = overall_time_end - overall_time_start
    train_time = train_end - train_start
    test_time = test_end - test_start
    vocab_time = vocab_end - vocab_start
    
    er = ExperimentResults(args.corpus, args.model, args.seed, args.hash, args.rare, args.ntrain, args.ntest, accuracy, all_time, train_time, test_time, vocab_time)

    with open(f'{args.corpus}_{args.model}_seed{args.seed}_hash{args.hash}_rare{args.rare}_train{args.ntrain}.pickle', 'wb') as save:
        pickle.dump(er, save)


if __name__ == '__main__':
    args = get_sysargs()
    run_experiment(args)

