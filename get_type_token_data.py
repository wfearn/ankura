import sys
import time
import gzip
import pickle
import os
import ankura
import psutil
import numpy as np
from copy import deepcopy
from collections import namedtuple
from ankura.heaps_utils import *
import argparse
import json
import os

def get_json_text(text, tag='reviewText'):
    line = json.loads(text)
    return line[tag]


def get_plaintext(text):
    return text


filext_dict = {
                'json' : get_json_text,
                'txt' : get_plaintext,
              }

def load(args):

    process = psutil.Process(os.getpid())
    ext = EXT_DICT[args.corpus]
    filename = get_corpus_file(args, ext=ext)
    print('Filename is:', filename)

    text_retriever = filext_dict[ext]

    l = list()
    with gzip.GzipFile(fileobj=open(filename, 'rb')) as f:
        for line in f:
            l.append(text_retriever(line.decode('utf-8')))

    type_token_dict = dict()
    t = ankura.pipeline.split_tokenizer()

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
                print(int(j / 1000000), 'million docs')
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
    print('Udreplace:', args.udreplace)
    print('Nremoval:', args.nremoval)
    print('Lower:', args.lower)
    print('Nopunct:', args.nopunct)
    load(args)
