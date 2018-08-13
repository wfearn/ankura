import sys
import pickle
import os
import ankura
import numpy as np
from copy import deepcopy
from collections import namedtuple

PlottingData = namedtuple('PlottingData', 'types tokens std')

def load(filename, sample_size, iterations):
    type_token_dict = dict()
    for i in range(iterations):
        print('Iteration', i)
        sys.stdout.flush()

        l = list()
        with open(filename, 'r') as f:
            for line in f:
                l.append(line)

        t = ankura.pipeline.default_tokenizer()

        print('Shuffling documents')
        sys.stdout.flush()
        np.random.shuffle(l)
        tokens = int(0)
        types = set()
        type_token_ratios = list()

        print('Getting ratio for sample size')
        sys.stdout.flush()
        for doc in l[:sample_size]:
            words = [w.token for w in t(doc)]
            tokens += len(words)
            types.update(set(words))
            type_token_ratios.append((len(types), tokens))

        type_token_dict[i] = deepcopy(type_token_ratios)
    
    final_type_token = list()
    for i in range(len(type_token_ratios)):
        avg_types = np.mean([type_token_dict[j][i][0] for j in range(iterations)])
        avg_tokens = np.mean([type_token_dict[j][i][1] for j in range(iterations)])
        std = np.std([type_token_dict[j][i][1] for j in range(iterations)])

        final_type_token.append(PlottingData(avg_types, avg_tokens, std))

    corpus_path, _ = os.path.splitext(filename)
    corpus_name = corpus_path.split('/')[-1]
    with open(f'{corpus_name}_{sample_size}_{iterations}.pickle', 'wb') as f:
        pickle.dump(final_type_token, f)


if __name__ == '__main__':
    filename = sys.argv[1]
    sample_size = int(sys.argv[2])
    iterations = int(sys.argv[3])
    load(filename, sample_size, iterations)
