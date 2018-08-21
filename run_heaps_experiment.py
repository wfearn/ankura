import ankura
from ankura.pipeline import HashedVocabBuilder
from ankura.pipeline import VocabBuilder
import json
import numpy as np
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from collections import namedtuple
from collections import defaultdict
import time
import os
import sys
import pickle
import argparse
from heaps_utils import *


def get_sysargs():
    parser = argparse.ArgumentParser(description='Process arguments for the experiment the user wants to run.')
    parser.add_argument('--corpus', type=str, required=True, help='the corpus the experiment will use', choices=['amazon_large', 'amazon_large_sample', 'na_news', 'twitter', 'amazon_large_corrected', 'amazon_large_test', 'yelp'])
    parser.add_argument('--model', type=str, required=True, help='ML model used by experiment', choices=['svm', 'nb', 'ankura'])
    parser.add_argument('--seed', type=int, default=0, nargs='?', help='seed for the random number generator')
    parser.add_argument('--hash', type=int, default=0, help='final hashed vocabulary size')
    parser.add_argument('--rare', type=int, default=0, help='rare word filter')
    parser.add_argument('--udreplace', dest='udreplace', action='store_true', help='whether to replace \'_\' and \'-\' with \' \'')
    parser.add_argument('--nremoval', dest='nremoval', action='store_true', help='whether to remove numbers')
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

def create_doc_topic_matrix(docs, num_topics, topics, theta_attr, label):
    doc_thetas = ankura.topic.gensim_assign(docs, topics, theta_attr=theta_attr)
    matrix = np.zeros((len(docs.documents), num_topics))

    for i, doc in enumerate(docs.documents):
        matrix[i, :] = np.log(doc_thetas[i] + 1e-30)

    target = [doc.metadata[label] for doc in docs.documents]

    return matrix, target


def run_experiment(args):
    overall_time_start = time.time()
    print('Running experiment')
    if args.model in SK_LEARN_MODELS: 

        corpus = get_sklearn_corpus(args.corpus)
        tokenizer = create_tokenizer(args)
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

        v = TfidfVectorizer(min_df=args.rare)

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
        split, test = ankura.pipeline.train_test_split(corpus, test_name=f'{args.corpus}_{args.model}_{args.ntest}_test', train_name=f'{args.corpus}_{args.model}_trainp1_split', random_seed=args.seed, num_train=args.ntrain + 1, num_test=args.ntest, vocab_size=args.hash)
        train, _ = ankura.pipeline.train_test_split(split, test_name=f'{args.corpus}_{args.model}_1_nan', train_name=f'{args.corpus}_{args.model}_{args.ntrain}_train', random_seed=args.seed, num_train=args.ntrain, num_test=1)


        doc_label = LABEL_DICT[args.corpus]
        vocab_start = time.time()
        Q = ankura.anchor.build_supervised_cooccurrence(train, doc_label, range(len(train.documents)))
        vocab_end = time.time()

        anchors = ankura.anchor.gram_schmidt_anchors(train, Q, NUM_TOPICS, doc_threshold=100)
        topics = ankura.anchor.recover_topics(Q, anchors, 1e-5)

        train_matrix, train_target = create_doc_topic_matrix(train, NUM_TOPICS, topics, 'z', doc_label)
        test_matrix, test_target = create_doc_topic_matrix(test, NUM_TOPICS, topics, 'z', doc_label)

        lr =  LogisticRegression()

        train_start = time.time()
        lr.fit(train_matrix, train_target)
        train_end = time.time()

        test_start = time.time()
        accuracy = lr.score(test_matrix, test_target)
        test_end = time.time()

    overall_time_end = time.time()

    all_time = overall_time_end - overall_time_start
    train_time = train_end - train_start
    test_time = test_end - test_start
    vocab_time = vocab_end - vocab_start

    vocabulary = len(vb.tokens) if args.model in SK_LEARN_MODELS else len(corpus.vocabulary)
    
    er = ExperimentResults(args.corpus, args.model, args.seed, args.hash, args.rare, args.ntrain, args.ntest, accuracy, vocabulary, all_time, train_time, test_time, vocab_time)

    print('Vocabulary Size:', vocabulary)

    with open(f'{args.corpus}_{args.model}_seed{args.seed}_hash{args.hash}_rare{args.rare}_train{args.ntrain}.pickle', 'wb') as save:
        pickle.dump(er, save)


if __name__ == '__main__':
    args = get_sysargs()
    run_experiment(args)
