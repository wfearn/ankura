import ankura
from copy import deepcopy
import random
import os
import pickle
import sys
import signal
import time
import numpy as np
import psutil
from collections import namedtuple

TTRatio = namedtuple('TTRatio', 'types tokens')
PlottingData = namedtuple('PlottingData', 'types tokens std')
corpus_dict = ankura.corpus.corpus_dictionary


def run(corpus_name='yelp', t_size=4000000, iters=5):

    type_set = set()
    tokens = int(0)
    types_to_tokens = list()
    type_token_dict = dict()
    print('Retrieving Corpus')
    corpus = corpus_dict[corpus_name]()    
    print('Corpus Retrieved')

    print('Getting types and tokens')
    operation_time = int(0)
    sys.stdout.flush()


    start = time.time()
    for j in range(iters):

        types_to_tokens.clear() 
        type_set.clear()
        tokens = 0

        print('Splitting Corpus')
        subset, _ = ankura.pipeline.train_test_split(corpus, num_train=t_size, num_test=1, train_name=f'{corpus_name}_{t_size}_train', test_name=f'{corpus_name}_{t_size}_test', save_dir='/fslhome/wfearn/compute/amazon_large')
        print('Splitting Done')
        sys.stdout.flush()

        #random.shuffle(corpus.documents)
        for i, doc in enumerate(subset.documents):
            tokens += len(doc.tokens)
            type_set.update(set([t.token for t in doc.tokens]))
            types = len(type_set)
            types_to_tokens.append(TTRatio(types, tokens))

            if i % 1000000 == 0 and i is not 0:
                sofar = time.time()
                time_so_far = (sofar - start)
                print('Time taken so far:', time_so_far)
                print('Average time:', time_so_far / i)
                sys.stdout.flush()
            
        type_token_dict[j] = deepcopy(types_to_tokens)

    sys.stdout.flush()
    end = time.time()
    total_time = end - start
    print('Total time:', total_time)
    sys.stdout.flush()

    final_type_token = list() 
    for i in range(len(types_to_tokens)):
        types = np.mean([type_token_dict[j][i].types for j in range(5)])
        tokens = np.mean([type_token_dict[j][i].tokens for j in range(5)])
        std = np.std([type_token_dict[j][i].types for j in range(5)])

        final_type_token.append(PlottingData(types, tokens, std))

    print('Types and tokens done, printing results')
    with open(f'/fslhome/wfearn/ankura/{corpus_name}_type_token.pickle', 'wb') as f:
        pickle.dump(final_type_token, f)

if __name__ == '__main__':
    corpus_name = sys.argv[1]
    training_size = int(sys.argv[2])
    iterations = int(sys.argv[3])
    run(corpus_name=corpus_name, t_size=training_size, iters=iterations)
