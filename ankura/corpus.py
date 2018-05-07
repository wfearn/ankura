"""Provides access to some standard downloadable datasets.

The available datasets (and corresponding import functions) include:
    * bible
    * newsgroups
    * amazon
These imports depend on two module variables which can be mutated to change the
download behavior of these imports. Downloaded and pickled data will be stored
in the path given by `download_dir`, and data will be downloaded from
`base_url`. By default, `download_dir` will be '$HOME/.ankura' while base_url
will point at a GitHub repo designed for use with 
"""
import functools
import itertools
import os
import urllib.request

from . import pipeline
import posixpath

download_dir = os.path.join(os.getenv('HOME'), '.ankura')

def _path(name):
    return os.path.join(download_dir, name)


base_url = 'https://github.com/wfearn/data/raw/data2'

def _url(name):
    return posixpath.join(base_url, name)


def _ensure_dir(path):
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass


def _ensure_download(name):
    path = _path(name)
    if not os.path.isfile(path):
        _ensure_dir(path)
        urllib.request.urlretrieve(_url(name), path)


def open_download(name, mode='r'):
    """Gets a file object for the given name, downloading the data to download
    dir from base_url if needed. By default the files are opened in read mode.
    If used as part of an inputer, the mode should likely be changed to binary
    mode. For a list of useable names with the default base_url, see
    download_inputer.
    """
    _ensure_download(name)
    return open(_path(name), mode)


def download_inputer(*names):
    """Generates file objects for the given names, downloading the data to
    download_dir from base_url if needed. Using the default base_url the
    available names are:
        * bible/bible.txt
        * bible/xref.txt
        * newsgroups/newsgroups.tar.gz
        * stopwords/english.txt
        * stopwords/jacobean.txt
    """
    @functools.wraps(download_inputer)
    def _inputer():
        for name in names:
            yield open_download(name, mode='rb')
    return _inputer

def tripadvisor():
    """Gets a corpus containing hotel reviews on trip advisor with ~240,000 documents"""

    label_stream = BufferedStream()

    def regex_extractor(docfile):
        import re

        text = docfile.read().decode('utf-8')

        documents = re.findall('<Content>(.*$)', text, re.M)
        labels = re.findall('<Overall>(\d*)', text, re.M)

        for i in range(len(documents)):

            overall = int(labels[i])
            label = overall if overall == 5 else 0
            label_stream.append(str(i), label)
            yield pipeline.Text(str(i), documents[i])


    p = pipeline.Pipeline(
            download_inputer('tripadvisor/tripadvisor.tar.gz'),
            pipeline.targz_extractor(regex_extractor),
            pipeline.stopword_tokenizer(
                    pipeline.default_tokenizer(),
                    open_download('stopwords/english.txt'),
            ),
            pipeline.stream_labeler(label_stream),
            pipeline.length_filterer(30),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 150, 20000)
    return p.run(_path('tripadvisor.pickle'))

def bible():
    """Gets a Corpus containing the King James version of the Bible with over
    250,000 cross references.
    """
    p= pipeline.Pipeline(
        download_inputer('bible/bible.txt'),
        pipeline.line_extractor(),
        pipeline.stopword_tokenizer(
            pipeline.default_tokenizer(),
            itertools.chain(
                open_download('stopwords/english.txt'),
                open_download('stopwords/jacobean.txt'),
            )
        ),
        pipeline.composite_labeler(
            pipeline.title_labeler('verse'),
            pipeline.list_labeler(
                open_download('bible/xref.txt'),
                'xref',
            ),
        ),
        pipeline.keep_filterer(),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 2)
    return p.run(_path('bible.pickle'))

def toy():
    p = pipeline.Pipeline(
        download_inputer('toy/toy.tar.gz'),
        pipeline.targz_extractor(
            pipeline.whole_extractor()
        ),
        pipeline.default_tokenizer(),
        pipeline.composite_labeler(
            pipeline.title_labeler('id'),
            pipeline.dir_labeler('directory')
        ),
        pipeline.length_filterer(),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p)
    return p.run(_path('toy.pickle'))

def newsgroups(hash_size=None, rare_word_filter=80, common_word_filter=2000, dir_prefix=None):
    """Gets a Corpus containing roughly 20,000 usenet postings from 20
    different newsgroups in the early 1990's.
    """

    base_file = 'newsgroups{}.pickle'
    base_file = base_file.format('{}_rare' + str(rare_word_filter) + '_common' + str(common_word_filter))
    base_file = base_file.format('_vocab' + str(hash_size)) if hash_size else base_file.format('')

    coarse_mapping = {
        'comp.graphics': 'comp',
        'comp.os.ms-windows.misc': 'comp',
        'comp.sys.ibm.pc.hardware': 'comp',
        'comp.sys.mac.hardware': 'comp',
        'comp.windows.x': 'comp',
        'rec.autos': 'rec',
        'rec.motorcycles': 'rec',
        'rec.sport.baseball': 'rec',
        'rec.sport.hockey': 'rec',
        'sci.crypt': 'sci',
        'sci.electronics': 'sci',
        'sci.med': 'sci',
        'sci.space': 'sci',
        'misc.forsale': 'misc',
        'talk.politics.misc': 'politics',
        'talk.politics.guns': 'politics',
        'talk.politics.mideast': 'politics',
        'talk.religion.misc' : 'religion',
        'alt.atheism' : 'religion',
        'soc.religion.christian' : 'religion',
    }

    p = pipeline.Pipeline(
        download_inputer('newsgroups/newsgroups.tar.gz'),
        pipeline.targz_extractor(
            pipeline.skip_extractor(errors='replace'),
        ),
        pipeline.remove_tokenizer(
            pipeline.stopword_tokenizer(
                pipeline.default_tokenizer(),
                itertools.chain(open_download('stopwords/english.txt'),
                                open_download('stopwords/newsgroups.txt'))
            ),
            r'^(.{0,2}|.{15,})$', # remove any token t for which 2<len(t)<=15
        ),
        pipeline.composite_labeler(
            pipeline.title_labeler('id'),
            pipeline.dir_labeler('newsgroup'),
            lambda n: {'coarse_newsgroup': coarse_mapping[os.path.dirname(n)]},
        ),
        pipeline.length_filterer(),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, rare_word_filter, common_word_filter)
    return p.run(_path(base_file), hash_size=hash_size)

def amazon_modified(corpus_size=1000000, vocab_size=None):
    """Gets a corpus containing a number Amazon product reviews 
    equal to corpus_size, with star ratings. """

    from .util import create_amazon_modified
    create_amazon_modified(corpus_size)

    label_stream = BufferedStream()

    def label_extractor(docfile, value_key='reviewText', label_key='overall'):

        import json

        for i, line in enumerate(docfile):
            line = json.loads(line.decode('utf-8'))
            label_stream.append(str(i), line[label_key])

            yield pipeline.Text(str(i), line[value_key])

    p = pipeline.Pipeline(
        download_inputer('amazon_modified/amazon_modified.json.gz'),
        pipeline.gzip_extractor(label_extractor),
        pipeline.stopword_tokenizer(
            pipeline.default_tokenizer(),
            open_download('stopwords/english.txt'),
        ),
        pipeline.stream_labeler(label_stream),
        pipeline.length_filterer(),
    )

    p.tokenizer = pipeline.frequency_tokenizer(p)
    return p.run(_path('amazon_modified.pickle'), hash_size=vocab_size)

def amazon_large(hash_size=0, rare_word_filter=80, common_word_filter=20000, dir_prefix=None):
    """Gets a corpus containing ~80 million Amazon product reviews, with star ratings.
    """

    pickle_file = 'amazon_large{}.corpus.pickle'
    pickle_file = pickle_file.format('{}_vocab' + str(hash_size)) if hash_size else pickle_file.format('{}')
    pickle_file = pickle_file.format('_rare' + str(rare_word_filter)) if rare_word_filter else pickle_file.format('')

    print('Pickle file name is:', pickle_file)

    label_stream = BufferedStream()

    def hingidy_jingidies(docfile, value_key='reviewText', label_key='overall'):

        import json

        for i, line in enumerate(docfile):
            line = json.loads(line.decode('utf-8'))    
            label_stream.append(str(i), line[label_key])

            yield pipeline.Text(str(i), line[value_key])

    p = pipeline.Pipeline(
        download_inputer('amazon_large/amazon_large.json.gz'),
        pipeline.gzip_extractor(hingidy_jingidies),
        pipeline.stopword_tokenizer(
            pipeline.default_tokenizer(),
            open_download('stopwords/english.txt'),
        ),
        pipeline.stream_labeler(label_stream),
        pipeline.length_filterer(),
    )

    doc_stream = dir_prefix + '/amazon_large{}.docs.pickle' if dir_prefix else '/var/tmp/amazon_large{}.docs.pickle'
    doc_stream = doc_stream.format('{}_vocab' + str(hash_size)) if hash_size else doc_stream.format('{}') 
    doc_stream = doc_stream.format('_rare' + str(rare_word_filter)) if rare_word_filter else doc_stream.format('')

    p.tokenizer = pipeline.frequency_tokenizer(p, rare=rare_word_filter) if rare_word_filter else pipeline.frequency_tokenizer(p)
    return p.run(_path(pickle_file), doc_stream, hash_size)

def amazon_medium():
    """Gets a corpus containing 100,000 Amazon product reviews, with star ratings.
    """
    label_stream = BufferedStream()

    def label_extractor(docfile, value_key='reviewText', label_key='overall'):

        import json

        for i, line in enumerate(docfile):
            line = json.loads(line.decode('utf-8'))
            label_stream.append(str(i), line[label_key])

            yield pipeline.Text(str(i), line[value_key])

    p = pipeline.Pipeline(
        download_inputer('amazon_medium/amazon_medium.json.gz'),
        pipeline.gzip_extractor(label_extractor),
        pipeline.stopword_tokenizer(
            pipeline.default_tokenizer(),
            open_download('stopwords/english.txt'),
        ),
        pipeline.stream_labeler(label_stream),
        pipeline.length_filterer(),
    )

    p.tokenizer = pipeline.frequency_tokenizer(p, 100, 2000)
    return p.run(_path('amazon_medium.pickle'))

def yelp():
    """Gets a corpus containing roughly 25,000 yelp restaurant reviews, with ratings."""

    binary_mapping = { # How do I use this?
            5 : 5,
            4 : 0,
            3 : 0,
            2 : 0,
            1 : 0
    }

    p = pipeline.Pipeline(
        download_inputer('yelp/yelp.txt'),
        pipeline.line_extractor('\t'),
        pipeline.stopword_tokenizer(
            pipeline.default_tokenizer(),
            open_download('stopwords/english.txt'),
        ),
        pipeline.composite_labeler(
            pipeline.title_labeler('id'),
            pipeline.float_labeler(
                open_download('yelp/yelp.response'),
                'rating',
            ),
        ),
        pipeline.length_filterer(),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p)
    return p.run(_path('yelp.pickle'))

def amazon():
    """Gets a Corpus containing roughly 40,000 Amazon product reviews, with
    star ratings.
    """
    p = pipeline.Pipeline(
        download_inputer('amazon/amazon.txt'),
        pipeline.line_extractor('\t'),
        pipeline.stopword_tokenizer(
            pipeline.default_tokenizer(),
            open_download('stopwords/english.txt'),
        ),
        pipeline.composite_labeler(
            pipeline.title_labeler('id'),
            pipeline.float_labeler(
                open_download('amazon/amazon.stars'),
                'rating',
            ),
        ),
        pipeline.length_filterer(30),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 50)
    return p.run(_path('amazon.pickle'))

class BufferedStream(object):

    def __init__(self):
        self.buf = []

    def append(self, key, value):
        self.buf.append((key, value))

    def __iter__(self):
        while self.buf:
            tup = self.buf.pop()
            key = tup[0]
            val = tup[1]

            yield key, val
