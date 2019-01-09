import os
import re
import sys
import glob
from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np

from functools import partial
from collections import namedtuple, defaultdict, Counter


from anikattu.tokenizer import word_tokenize
from anikattu.tokenstring import TokenString
from anikattu.trainer.lm import Trainer , Tester, Predictor
from anikattu.datafeed import DataFeed, MultiplexedDataFeed
from anikattu.dataset import NLPDataset as Dataset, NLPDatasetList as DatasetList
from anikattu.utilz import tqdm, ListTable
from anikattu.vocab import Vocab
from anikattu.utilz import Var, LongVar, init_hidden, pad_seq
from nltk.tokenize import WordPunctTokenizer
word_punct_tokenizer = WordPunctTokenizer()
word_tokenize = word_punct_tokenizer.tokenize

from tace16.tace16 import tace16_to_utf8, utf8_to_tace16

VOCAB =  ['PAD', 'UNK', 'GO', 'EOS']
PAD = VOCAB.index('PAD')

"""
    Local Utilities, Helper Functions

"""
Sample   =  namedtuple('Sample', ['id', 'sequence', 'char_level'])

def __repr__(self):
    if self.char_level:
        s = tace16_to_utf8([int(i) for i in self.sequence])
    else:
        s = self.sequence
        
    return '<{}:{}, {}>'.format(
        self.__class__.__name__,
        self.id,
        s
    )
     

Sample.__repr__ = __repr__

def unicodeToAscii(s):
    import unicodedata
    return ''.join(
        c for c in unicodedata.normalize('NFKC', s)
        if unicodedata.category(c) != 'Mn'
    )

def load_tawiki_data(config, dataset_name='tawiki', char_level = True, max_sample_size=None):
    samples = []
    skipped = 0

    vocab = Counter()
    
    try:
        filename = glob.glob('../dataset/tawiki_lines.txt')[0]
              
        log.info('processing file: {}'.format(filename))
        dataset = open(filename).readlines()
        for i, line in enumerate(tqdm(dataset, desc='processing {}'.format(filename))):
            import string

            #print(line)
            try:
                line = line.strip()
                if len(line) > 0:
                    if char_level:
                        """
                        for j, word in enumerate(line.split()):
                            word = ''.join([i for i in word if i not in string.printable or i == ' '])
                            samples.append(
                                Sample(
                                    id = '{}.{}.{}'.format(dataset_name, i ,j),
                                    sequence = [str(i) for i in utf8_to_tace16(word)],
                                    char_level = True
                                )
                            )
                        """
                        samples.append(
                            Sample(
                                id = '{}.{}'.format(dataset_name, i),
                                sequence = [str(i) for i in utf8_to_tace16(line)],
                                char_level = True
                            )
                        )
                    else:
                        samples.append(
                            Sample(
                                id = '{}.{}'.format(dataset_name, i),
                                sequence = line.split(),
                                char_level = False
                            )
                        )
            except:
                log.exception('{}.{} -  {}'.format(dataset_name, i, line))
    except:
        skipped += 1
        log.exception('{}.{} -  {}'.format(dataset_name, i, line))

    print('skipped {} samples'.format(skipped))
    
    samples = sorted(samples, key=lambda x: len(x.sequence), reverse=True)
    if max_sample_size:
        samples = samples[:max_sample_size]

    log.info('building vocab...')
    for sample in samples:
        vocab.update(sample.sequence)

    return os.path.basename(filename), samples, vocab

def load_tawiki_bpe_data(config, dataset_name='tawiki', max_sample_size=None):
    samples = []
    skipped = 0

    vocab = Counter()
    
    try:
        filename = glob.glob('../dataset/tawiki_lines_bpe.txt')[0]
              
        log.info('processing file: {}'.format(filename))
        dataset = open(filename).readlines()
        for i, line in enumerate(tqdm(dataset, desc='processing {}'.format(filename))):
            import string

            #print(line)
            try:
                line = line.strip()
                if len(line) > 0:
                    samples.append(
                        Sample(
                            id = '{}.{}'.format(dataset_name, i),
                            sequence = line.split(),
                            char_level = False
                        )
                    )
            except:
                log.exception('{}.{} -  {}'.format(dataset_name, i, line))
    except:
        skipped += 1
        log.exception('{}.{} -  {}'.format(dataset_name, i, line))

    print('skipped {} samples'.format(skipped))
    
    samples = sorted(samples, key=lambda x: len(x.sequence), reverse=True)
    if max_sample_size:
        samples = samples[:max_sample_size]

    log.info('building vocab...')
    for sample in samples:
        vocab.update(sample.sequence)

    return os.path.basename(filename), samples, vocab


def load_data(config, max_sample_size=None, char_level=True):
    dataset = {}
    #filename, samples, vocab = load_tawiki_data(config, char_level=char_level)
    filename, samples, vocab = load_tawiki_bpe_data(config)
    vocab = Vocab(vocab, special_tokens=VOCAB)
    pivot = int( config.CONFIG.split_ratio * len(samples))
    train_samples, test_samples = samples[:pivot], samples[pivot:]
    dataset[filename] = Dataset(filename, (train_samples, test_samples), vocab, vocab)

    return DatasetList('ta-lm', dataset.values())
        

# ## Loss and accuracy function
def loss(ti, output, batch, loss_function, *args, **kwargs):
    indices, (sequence, ), _ = batch
    output, state = output
    return loss_function(output, sequence[:, ti+1]) 



def accuracy(ti, output, batch, *args, **kwargs):
    indices, (sequence, ), _ = batch
    output, state = output
    return (output.max(dim=1)[1] == sequence[:, ti+1]).sum().float()/float(answer.size(0))/float(answer.size(1))


def repr_function(output, batch, VOCAB, dataset):
    indices, (sequence, ), _ = batch
    results = []
    for idx, o in zip(indices, output):
        o = ' '.join([VOCAB[o]])
        
    return results


def batchop(datapoints, VOCAB, config, *args, **kwargs):
    indices = [d.id for d in datapoints]
    sequence = []
    for d in datapoints:
        s = []
        sequence.append([VOCAB[w] for w in d.sequence])

    sequence    = LongVar(config, pad_seq(sequence))
    batch = indices, (sequence, ), ()
    return batch
    

def portion(dataset, percent):
    return dataset[ : int(len(dataset) * percent) ]


def train(config, argv, name, ROOT_DIR,  model, dataset):
    _batchop = partial(batchop, VOCAB=dataset.input_vocab, config=config)
    predictor_feed = DataFeed(name,
                              dataset.testset,
                              batchop = _batchop,
                              batch_size=1)
    
    train_feed     = DataFeed(name,
                              portion(dataset.trainset,
                                      config.HPCONFIG.trainset_size),
                              batchop    = _batchop,
                              batch_size = config.CONFIG.batch_size)
    
    
    loss_ = partial(loss, loss_function=nn.NLLLoss())
    test_feed, tester = {}, {}
    for subset in dataset.datasets:
        test_feed[subset.name]      = DataFeed(subset.name,
                                               subset.testset,
                                               batchop    = _batchop,
                                               batch_size = config.CONFIG.batch_size)

        tester[subset.name] = Tester(name     = subset.name,
                                     config   = config,
                                     model    = model,
                                     directory = ROOT_DIR,
                                     loss_function = loss_,
                                     accuracy_function = loss_,
                                     feed = test_feed[subset.name],
                                     save_model_weights=False)

    test_feed[name]      = DataFeed(name, dataset.testset, batchop=_batchop, batch_size=config.CONFIG.batch_size)

    tester[name] = Tester(name  = name,
                          config   = config,
                          model    = model,
                          directory = ROOT_DIR,
                          loss_function = loss_,
                          accuracy_function = loss_,
                          feed = test_feed[name],
                          predictor=predictor)
    
    
    def do_every_checkpoint(epoch):
        if config.CONFIG.plot_metrics:
            from matplotlib import pyplot as plt
            fig = plt.figure(figsize=(10, 5))
            
        for t in tester.values():
            t.do_every_checkpoint(epoch)

            if config.CONFIG.plot_metrics:
                plt.plot(list(t.loss), label=t.name)

        if config.CONFIG.plot_metrics:
            plt.savefig('loss.png')
            plt.close()
        
            
            
    for e in range(config.CONFIG.EONS):
        if not trainer.train():
            raise Exception
