"""Provides access to some standard downloadable datasets.

The available datasets (and corresponding import functions) include:
    * bible
    * newsgroups
    * amazon
These imports depend on two module variables which can be mutated to change the
download behavior of these imports. Downloaded and pickled data will be stored
in the path given by `download_dir`, and data will be downloaded from
`base_url`. By default, `download_dir` will be '$HOME/.ankura' while base_urlwill point at a GitHub repo designed for use with
"""
import functools
import itertools
import os
import urllib.request
import sys

from . import pipeline
import posixpath

download_dir = os.path.join(os.getenv('HOME'), 'compute/.ankura')
base_url = 'https://github.com/wfearn/data/raw/data2'


def _path(name):
    return os.path.join(download_dir, name)


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


def _binary_labeler(data, threshold,
                    attr='label', delim='\t',
                    needs_split=True):
    stream = data
    if needs_split:
        stream = (line.rstrip(os.linesep).split(delim, 1) for line in stream)
    stream = ((key, float(value) >= threshold) for key, value in stream)
    return pipeline.stream_labeler(stream, attr)


def _binary_string_labeler(data, threshold,
                           attr='binary_rating', delim='\t',
                           needs_split=True):
    stream = data
    if needs_split:
        stream = (line.rstrip(os.linesep).split(delim, 1) for line in stream)
    stream = ((key, 'positive' if float(value) >= threshold else 'negative')
               for key, value in stream)
    return pipeline.stream_labeler(stream, attr)


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
    #These words should be removed, as they don't en up cooccuring with other
    # words
    extra_stopwords = ['zentral', 'jederzeit', 'gerne', 'gutes', 'preis',
    'plac茅', 'empfehlenswert', 'preisleistungsverh盲ltnis', 'posizione',
    'sch枚nes', 'zu', 'empfehlen', 'qualit茅prix', 'tolles', 'rapporto',
    'guter', 'struttura']

    label_stream = []

    def regex_extractor(docfile):
        import re

        text = docfile.read().decode('utf-8')

        documents = re.findall('<Content>(.*$)', text, re.M)
        labels = re.findall('<Overall>(\d*)', text, re.M)

        for i in range(len(documents)):
            overall = int(labels[i])
            label = overall if overall == 5 else 0
            label_stream.append((str(i), label))
            yield pipeline.Text(str(i), documents[i])

    p = pipeline.Pipeline(
        download_inputer('tripadvisor/tripadvisor.tar.gz'),
        pipeline.targz_extractor(regex_extractor),
        pipeline.stopword_tokenizer(
                pipeline.default_tokenizer(),
                itertools.chain(
                    open_download('stopwords/english.txt'),
                    extra_stopwords
                )
        ),
        pipeline.stream_labeler(label_stream),
        pipeline.length_filterer(30),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 150, 20000)
    return p.run(_path('tripadvisor.pickle'))


def yelp():
    """ Gets a corpus containing Yelp reviews with 25431 documents """

    base_tokenzer = pipeline.default_tokenizer()

    def strip_non_alpha(token):
        return ''.join(c for c in token if c.isalnum())


    def tokenizer(data):
        tokens = base_tokenzer(data)
        tokens = [pipeline.TokenLoc(strip_non_alpha(t.token), t.loc) for t in tokens]
        tokens = [t for t in tokens if t.token]
        return tokens


    def binary_labeler(data, threshold, attr='label', delim='\t'):
        stream = (line.rstrip(os.linesep).split(delim, 1) for line in data)
        stream = ((key, float(value) >= threshold) for key, value in stream)
        return pipeline.stream_labeler(stream, attr)


    p = pipeline.Pipeline(
        download_inputer('yelp/yelp.txt'),
        pipeline.line_extractor('\t'),
        tokenizer,
        pipeline.composite_labeler(
            pipeline.title_labeler('id'),
            pipeline.float_labeler(
                open_download('yelp/yelp.response'),
                'rating',
            ),
            _binary_string_labeler(
                open_download('yelp/yelp.response'),
                5,
                'binary_rating',
            ),
        ),
        pipeline.length_filterer(30),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 50)
    return p.run(_path('yelp.pickle'))


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


def nyt():
    label_stream = BufferedStream()

    def stream_extractor(docfile, label_key='nyt'):
        for i, line in enumerate(docfile):
            line = line.decode('utf-8')
            label_stream.append((str(i), label_key))
            yield pipeline.Text(str(i), line)

    p = pipeline.Pipeline(
        download_inputer('nyt/nyt.txt.gz'),
        pipeline.gzip_extractor(stream_extractor),
        pipeline.default_tokenizer(),
        pipeline.stream_labeler(label_stream),
        pipeline.length_filterer(),
    )

    p.tokenizer = pipeline.frequency_tokenizer(p)
    return p.run(_path('nyt.pickle'), docs_path=_path('nyt.docs.pickle'))


def newsgroups():
    """Gets a Corpus containing roughly 20,000 usenet postings from 20
    different newsgroups in the early 1990's.
    """
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
            r'^(.{0,2}|.{15,})$', # remove any token t with len(t)<=2 or len(t)>=15
        ),
        pipeline.composite_labeler(
            pipeline.title_labeler('id'),
            pipeline.dir_labeler('newsgroup'),
            lambda n: {'coarse_newsgroup': coarse_mapping[os.path.dirname(n)]},
        ),
        pipeline.length_filterer(),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p)
    return p.run(_path('newsgroups.pickle'))


def amazon_modified(corpus_size=10000000, vocab_size=None, rare=100, run_number=0):
    """Gets a corpus containing a number Amazon product reviews 
    equal to corpus_size, with star ratings. """

    from .util import create_amazon_modified
    create_amazon_modified(corpus_size, run_number)

    label_stream = BufferedStream()

    def label_extractor(docfile, value_key='reviewText', label_key='overall'):

        import json

        for i, line in enumerate(docfile):
            line = json.loads(line.decode('utf-8'))
            label_stream.append(str(i), line[label_key])

            yield pipeline.Text(str(i), line[value_key])

    p = pipeline.Pipeline(
        download_inputer(f'amazon_modified/amazon_modified_{corpus_size}_{run_number}.json.gz'),
        pipeline.gzip_extractor(label_extractor),
        pipeline.stopword_tokenizer(
            pipeline.default_tokenizer()
        ),
        pipeline.stream_labeler(label_stream),
        pipeline.length_filterer(),
    )

    p.tokenizer = pipeline.frequency_tokenizer(p, rare=rare)

    corpus_path = _path(f'amazon_modified_{corpus_size}_{run_number}.pickle')
    docs_path = _path(f'amazon_modified_{corpus_size}_{run_number}.docs.pickle')
    return corpus_path, docs_path, p.run(corpus_path, hash_size=vocab_size, docs_path=docs_path)


def wikipedia_corrected():
    """Gets a corpus containing ~80 million Amazon product reviews, with star ratings.
    """
    label_stream = BufferedStream()

    def stream_extractor(docfile, label_key='nyt'):
        for i, line in enumerate(docfile):
            line = line.decode('utf-8')
            label_stream.append(str(i), label_key)
            yield pipeline.Text(str(i), line)

    p = pipeline.Pipeline(
        download_inputer('wikipedia/wikipedia_corrected.txt.gz'),
        pipeline.gzip_extractor(stream_extractor),
        pipeline.default_tokenizer(),
        pipeline.stream_labeler(label_stream),
        pipeline.length_filterer(),
    )

    p.tokenizer = pipeline.frequency_tokenizer(p)
    return p.run(_path('wikipedia_corrected.pickle'))


def wikipedia():
    """Gets a corpus containing ~80 million Amazon product reviews, with star ratings.
    """
    label_stream = BufferedStream()

    def stream_extractor(docfile, label_key='nyt'):
        for i, line in enumerate(docfile):
            line = line.decode('utf-8')
            label_stream.append(str(i), label_key)
            yield pipeline.Text(str(i), line)

    p = pipeline.Pipeline(
        download_inputer('wikipedia/wikipedia.txt.gz'),
        pipeline.gzip_extractor(stream_extractor),
        pipeline.default_tokenizer(),
        pipeline.stream_labeler(label_stream),
        pipeline.length_filterer(),
    )

    p.tokenizer = pipeline.frequency_tokenizer(p)
    return p.run(_path('wikipedia.pickle'))


def nyt_corrected():
    """Gets a corpus containing ~80 million Amazon product reviews, with star ratings.
    """
    label_stream = BufferedStream()

    def stream_extractor(docfile, label_key='nyt'):
        for i, line in enumerate(docfile):
            line = line.decode('utf-8')
            label_stream.append((str(i), label_key))
            yield pipeline.Text(str(i), line)

    p = pipeline.Pipeline(
        download_inputer('nyt/nyt_corrected.txt.gz'),
        pipeline.gzip_extractor(stream_extractor),
        pipeline.default_tokenizer(),
        pipeline.stream_labeler(label_stream),
        pipeline.length_filterer(),
    )

    p.tokenizer = pipeline.frequency_tokenizer(p)
    return p.run(_path('nyt_corrected.pickle'), docs_path=_path('nyt_corrected.docs.pickle'))


def amazon_corrected():
    """Gets a corpus containing ~80 million Amazon product reviews, with star ratings.
    """

    label_stream = BufferedStream()

    def stream_extractor(docfile, value_key='reviewText', label_key='overall'):

        import json

        for i, line in enumerate(docfile):
            try:
                line = json.loads(line.decode('utf-8'))    
                label_stream.append(str(i), line[label_key])
            except json.decoder.JSONDecodeError as e:
                print('Error', e)
                print('Offending line:', line)
                sys.stdout.flush()
                sys.exit(0)

            yield pipeline.Text(str(i), line[value_key])

    p = pipeline.Pipeline(
        download_inputer('amazon_corrected/amazon_corrected.json.gz'),
        pipeline.gzip_extractor(stream_extractor),
        pipeline.default_tokenizer(),
        pipeline.stream_labeler(label_stream),
        pipeline.length_filterer(),
    )

    p.tokenizer = pipeline.frequency_tokenizer(p)
    return p.run(_path('amazon_corrected.pickle'), docs_path='/fslhome/wfearn/compute/amazon_large/amazon_large_corpora/amazon_corrected.docs.pickle')

def amazon_large_noreplace(dir_prefix='/fslhome/wfearn/compute/amazon_large/amazon_large_corpora/'):
    """Gets a corpus containing ~80 million Amazon product reviews, with star ratings.
    """

    label_stream = BufferedStream()

    def stream_extractor(docfile, value_key='reviewText', label_key='overall'):

        import json

        for i, line in enumerate(docfile):
            line = json.loads(line.decode('utf-8'))
            label_stream.append((str(i), line[label_key]))

            yield pipeline.Text(str(i), line[value_key])

    p = pipeline.Pipeline(
        download_inputer('amazon_large/amazon_large.json.gz'),
        pipeline.gzip_extractor(stream_extractor),
        pipeline.default_tokenizer(replace=False),
        pipeline.stream_labeler(label_stream),
        pipeline.length_filterer(),
    )

    p.tokenizer = pipeline.frequency_tokenizer(p)
    return p.run(_path('amazon_large_noreplace.pickle'), docs_path=f'{dir_prefix}amazon_large_noreplace.docs.pickle')


def amazon_large(hash_size=0, dir_prefix='/local/amazon_large'):
    """Gets a corpus containing ~80 million Amazon product reviews, with star ratings.
    """

    label_stream = BufferedStream()

    def stream_extractor(docfile, value_key='reviewText', label_key='overall'):

        import json

        for i, line in enumerate(docfile):
            line = json.loads(line.decode('utf-8'))
            label_stream.append((str(i), line[label_key]))

            yield pipeline.Text(str(i), line[value_key])

    p = pipeline.Pipeline(
        download_inputer('amazon_large/amazon_large.json.gz'),
        pipeline.gzip_extractor(stream_extractor),
        pipeline.default_tokenizer(),
        pipeline.stream_labeler(label_stream),
        pipeline.length_filterer(),
    )

    p.tokenizer = pipeline.frequency_tokenizer(p)
    return p.run(_path('amazon_large.pickle'), docs_path='/fslhome/wfearn/compute/amazon_large/amazon_large_corpora/amazon_large.docs.pickle')


def amazon_medium():
    """Gets a corpus containing 100,000 Amazon product reviews, with star ratings.
    """
    label_stream = BufferedStream()

    def label_extractor(docfile, value_key='reviewText', label_key='overall'):

        import json

        for i, line in enumerate(docfile):
            line = json.loads(line.decode('utf-8'))
            label_stream.append((str(i), line[label_key]))

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


def amazon():
    """Gets a Corpus containing roughly 40,000 Amazon product reviews, with
    star ratings.
    """
    def binary_labeler(data, threshold, attr='label', delim='\t'):
        stream = (line.rstrip(os.linesep).split(delim, 1) for line in data)
        stream = ((key, float(value) >= threshold) for key, value in stream)
        return pipeline.stream_labeler(stream, attr)

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
            _binary_string_labeler(
                open_download('amazon/amazon.stars'),
                5,
                'binary_rating',
            ),
        ),
        pipeline.length_filterer(30),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 50)
    return p.run(_path('amazon.pickle'))


class BufferedStream(object):

    def __init__(self):
        self.buf = []

    def append(self, key_value):
        self.buf.append(key_value)

    def __iter__(self):
        while self.buf:
            tup = self.buf.pop()
            key = tup[0]
            val = tup[1]
            yield key, val

corpus_dictionary = {
                        'amazon_corrected' : amazon_corrected,
                        'amazon_large' : amazon_large,
                        'nyt' : nyt,
                        'nyt_corrected' : nyt_corrected,
                        'wikipedia' : wikipedia,
                        'wikipedia_corrected' : wikipedia_corrected,
                        'yelp' : yelp,
                    }
