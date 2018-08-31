import sys
import time
import pickle
import os
import ankura
import numpy as np
from copy import deepcopy
from collections import namedtuple
from ankura.heaps_utils import *
import argparse

def get_sysargs():
    parser = argparse.ArgumentParser(description='Process arguments for type token calculation the user wants to run')
    parser.add_argument('--corpus', type=str, required=True, help='corpus the calculation will use', choices=['amazon_large', 'amazon_large_sample', 'amazon_large_corrected', 'amazon_large_symspell_nopunct', 'amazon_large_symspell_udreplace', 'amazon_large_symspell', 'amazon_large_symspell_wordseg', 'na_news', 'na_news_symspell', 'na_news_symspell_wordseg'])
    parser.add_argument('--iterations', type=int, default=5, help='Number of times to filter through corpus and take sample')
    parser.add_argument('--sample_size', type=int, required=True, help='Size of sample to take from corpus')
    parser.add_argument('--udreplace', dest='udreplace', action='store_true', help='whether to replace \'_\' and \'-\' with \' \'')
    parser.add_argument('--nopunct', dest='nopunct', action='store_true', help='whether to remove punctuation')
    parser.add_argument('--lower', dest='lower', action='store_true', help='whether to lowercase the tokens')
    parser.add_argument('--nremoval', dest='nremoval', action='store_true', help='whether to remove numbers')
    return parser.parse_args()

def load(args):

    filename = get_corpus_file(args)
    print('Filename is:', filename)

    l = list()
    with open(filename, 'r') as f:
        for line in f:
            l.append(line)

    type_token_dict = dict()
    t = create_tokenizer(args)

    for i in range(args.iterations):
        print('Iteration', i)
        sys.stdout.flush()

        print('Shuffling documents')
        sys.stdout.flush()

        np.random.shuffle(l)
        tokens = int(0)
        types = set()
        type_token_ratios = list()

        step_size = (args.sample_size / 10000) #Get a type token count every .01%

        print('Getting ratio for sample size')
        sys.stdout.flush()
        start = time.time()
        for j, doc in enumerate(l[:args.sample_size]):
            if not j % 1000000 and j:
                intermediary = time.time()
                time_until_now = intermediary - start
                print(int(j/1000000), 'million docs')
                print('Time until now:', time_until_now)
                print('Average time:', (time_until_now / j))
                sys.stdout.flush()

            words = [w.token for w in t(doc)]
            tokens += len(words)
            types.update(set(words))
            
            if not j % step_size:
                type_token_ratios.append((len(types), tokens))

        type_token_dict[i] = deepcopy(type_token_ratios)
    
    print('Finished, collecting average data and standard deviation')
    sys.stdout.flush()

    final_type_token = list()
    for i in range(len(type_token_ratios)):
        avg_types = np.mean([type_token_dict[j][i][0] for j in range(args.iterations)])
        avg_tokens = np.mean([type_token_dict[j][i][1] for j in range(args.iterations)])
        std = np.std([type_token_dict[j][i][1] for j in range(args.iterations)])

        final_type_token.append(PlottingData(avg_types, avg_tokens, std))

    corpus_path, _ = os.path.splitext(filename)
    corpus_name = corpus_path.split('/')[-1]
    dump_file = f'{corpus_name}'
    dump_file += '_udreplace' if args.udreplace else ''
    dump_file += '_nremoval' if args.nremoval else ''
    dump_file += '_lower' if args.lower else ''
    dump_file += '_nopunct' if args.nopunct else ''
    with open(f'{dump_file}_{args.sample_size}_{args.iterations}.pickle', 'wb') as f:
        pickle.dump(final_type_token, f)


if __name__ == '__main__':
    args = get_sysargs()
    load(args)
