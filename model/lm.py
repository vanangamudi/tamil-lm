import os
import re
import sys
import json
import time
import random
from pprint import pprint, pformat

sys.path.append('..')
import config
from anikattu.logger import CMDFilter
import logging
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(name)s.%(funcName)s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from functools import partial


import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

from anikattu.utilz import Var, LongVar, init_hidden, EpochAverager, FLAGS, tqdm
from anikattu.debug import memory_consumed

class Base(nn.Module):
    def __init__(self, config, name):
        super(Base, self).__init__()
        self._name = name
        self.log = logging.getLogger(self._name)
        size_log_name = '{}.{}'.format(self._name, 'size')
        self.log.info('constructing logger: {}'.format(size_log_name))
        self.size_log = logging.getLogger(size_log_name)
        self.size_log.info('size_log')
        self.log.setLevel(config.CONFIG.LOG.MODEL.level)
        self.size_log.setLevel(config.CONFIG.LOG.MODEL.level)
        self.print_instance = 0
        
    def __(self, tensor, name='', print_instance=False):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            for i in range(len(tensor)):
                self.__(tensor[i], '{}[{}]'.format(name, i))
        else:
            self.size_log.debug('{} -> {}'.format(name, tensor.size()))
            if self.print_instance or print_instance:
                self.size_log.debug(tensor)

            
        return tensor

    def name(self, n=''):
        if n:
            return '{}.{}'.format(self._name, n)
        else:
            return self._name

        
    def loss_trend(self, total_count=10):
        if len(self.test_loss) > 4:
            losses = self.test_loss[-4:]
            count = 0
            for l, r in zip(losses, losses[1:]):
                if l < r:
                    count += 1
                    
            if count > total_count:
                return FLAGS.STOP_TRAINING

        return FLAGS.CONTINUE_TRAINING



class LM(Base):
    def __init__(self,
                 # config and name
                 config, name,

                 # model parameters
                 vocab_size,

                
                 # feeds
                 train_feed,
                 test_feed,

                 # loss function
                 loss_function,
                 accuracy_function=None,

                 f1score_function=None,
                 save_model_weights=True,
                 epochs = 1000,
                 checkpoint = 1,
                 early_stopping = True,

                 # optimizer
                 optimizer = None,
                 
    ):
        super().__init__(config, name)
        self.config = config
        self.embed_size = config.HPCONFIG.embed_size
        self.hidden_size = config.HPCONFIG.hidden_size
        self.vocab_size = vocab_size
        self.loss_function = loss_function
        
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        
        self.lm  = nn.GRUCell(self.embed.embedding_dim, self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.answer = nn.Linear(self.hidden_size, self.vocab_size)

        self.loss_function = loss_function if loss_function else nn.NLLLoss()
        self.accuracy_function = accuracy_function if accuracy_function else loss_function

        self.optimizer = optimizer if optimizer else optim.SGD(self.parameters(),
                                                               lr=0.01, momentum=0.1)
        
        self.f1score_function = f1score_function
        
        self.epochs = epochs
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping
        
        self.train_feed = train_feed
        self.test_feed = test_feed
        
        self.__build_stats()

        ########################################################################################
        #  Saving model weights
        ########################################################################################
        self.save_model_weights = save_model_weights
        self.best_model = (0.000001, self.cpu().state_dict())
        try:
            f = '{}/{}_best_model_accuracy.txt'.format(self.config.ROOT_DIR, self.name())
            if os.path.isfile(f):
                self.best_model = (float(open(f).read().strip()), self.cpu().state_dict())
                self.log.info('loaded last best accuracy: {}'.format(self.best_model[0]))
        except:
            log.exception('no last best model')

                        
        self.best_model_criteria = self.accuracy
        self.save_best_model()

        self.restore_checkpoint()
        if config.CONFIG.cuda:
             self.cuda()


    def restore_checkpoint(self):
        try:
            self.snapshot_path = '{}/weights/{}.{}'.format(self.config.ROOT_DIR, self.name(), 'pth')
            self.load_state_dict(torch.load(self.snapshot_path))
            log.info('loaded the old image for the model from :{}'.format(self.snapshot_path))
        except:
            log.exception('failed to load the model  from :{}'.format(self.snapshot_path))

    def __build_stats(self):
        ########################################################################################
        #  Saving model weights
        ########################################################################################
        
        # necessary metrics
        self.mfile_prefix = '{}/results/metrics/{}'.format(self.config.ROOT_DIR, self.name())
        self.train_loss  = EpochAverager(self.config,
                                       filename = '{}.{}'.format(self.mfile_prefix,   'train_loss'))
        
        self.test_loss  = EpochAverager(self.config,
                                        filename = '{}.{}'.format(self.mfile_prefix,   'test_loss'))
        self.accuracy   = EpochAverager(self.config,
                                        filename = '{}.{}'.format(self.mfile_prefix,  'accuracy'))
        
        self.metrics = [self.train_loss, self.test_loss, self.accuracy]
        # optional metrics
        if getattr(self, 'f1score_function'):
            self.tp = EpochAverager(self.config, filename = '{}.{}'.format(self.mfile_prefix,   'tp'))
            self.fp = EpochAverager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'fp'))
            self.fn = EpochAverager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'fn'))
            self.tn = EpochAverager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'tn'))
            
            self.precision = EpochAverager(self.config,
                                           filename = '{}.{}'.format(self.mfile_prefix,  'precision'))
            self.recall    = EpochAverager(self.config,
                                           filename = '{}.{}'.format(self.mfile_prefix,  'recall'))
            self.f1score   = EpochAverager(self.config,
                                           filename = '{}.{}'.format(self.mfile_prefix,  'f1score'))
          
            self.metrics += [self.tp, self.fp, self.fn, self.tn, self.precision, self.recall, self.f1score]
            
    def save_best_model(self):
        with open('{}/{}_best_model_accuracy.txt'.format(self.config.ROOT_DIR, self.name()), 'w') as f:
            f.write(str(self.best_model[0]))

        if self.save_model_weights:
            self.log.info('saving the last best model with accuracy {}...'.format(self.best_model[0]))

            torch.save(self.best_model[1],
                       '{}/weights/{:0.4f}.{}'.format(self.config.ROOT_DIR, self.best_model[0], 'pth'))
            
            torch.save(self.best_model[1],
                       '{}/weights/{}.{}'.format(self.config.ROOT_DIR, self.name(), 'pth'))

            
    def initial_hidden(self, batch_size):
        ret = Variable(torch.zeros( batch_size, self.lm.hidden_size))
        ret = ret.cuda() if self.config.CONFIG.cuda else ret
        return ret
    
    def forward(self, input_, state):
        input_emb  = self.__( self.embed(input_),  'input_emb')
        state = self.lm(input_emb, state)
        return F.log_softmax(self.answer(state), dim=-1), state
        
        
    def do_train(self):
        for epoch in range(self.epochs):
            self.log.critical('memory consumed : {}'.format(memory_consumed()))            
            self.epoch = epoch
            if epoch % max(1, (self.checkpoint - 1)) == 0:
                if self.do_validate() == FLAGS.STOP_TRAINING:
                    self.log.info('loss trend suggests to stop training')
                    return
                           
            self.train()
            for j in tqdm(range(self.train_feed.num_batch), desc='Trainer.{}'.format(self.name())):
                input_ = self.train_feed.next_batch()
                idxs, inputs, targets = input_
                sequence = inputs[0].transpose(0,1)
                _, batch_size = sequence.size()

                state = self.initial_hidden(batch_size)
                loss = 0
                output = sequence[0]
                for ti in range(1, sequence.size(0) - 1):
                    output = self.forward(output, state)
                    loss += self.loss_function(ti, output, input_)
                    output, state = output
                    output = output.max(1)[1]
                    
                loss.backward()
                self.train_loss.cache(loss.data.item())
                self.optimizer.step()


            self.log.info('-- {} -- loss: {}\n'.format(epoch, self.train_loss.epoch_cache))
            self.train_loss.clear_cache()
            
            for m in self.metrics:
                m.write_to_file()

        return True

    def do_validate(self):
        self.eval()
        for j in tqdm(range(self.test_feed.num_batch), desc='Tester.{}'.format(self.name())):
            input_ = self.test_feed.next_batch()
            idxs, inputs, targets = input_
            sequence = inputs[0].transpose(0,1)
            _, batch_size = sequence.size()
            
            state = self.initial_hidden(batch_size)
            loss, accuracy = Var(self.config, [0]), Var(self.config, [0])
            output = sequence[0]
            outputs = []
            for ti in range(1, sequence.size(0) - 1):
                output = self.forward(output, state)
                loss += self.loss_function(ti, output, input_)
                accuracy += self.accuracy_function(ti, output, input_)
                output, state = output
                output = output.max(1)[1]
                outputs.append(output)
                
            self.test_loss.cache(loss.item())
            if ti == 0: ti = 1
            self.accuracy.cache(accuracy.item()/ti)
            #print('====', self.test_loss, self.accuracy)

        self.log.info('= {} =loss:{}'.format(self.epoch, self.test_loss.epoch_cache))
        self.log.info('- {} -accuracy:{}'.format(self.epoch, self.accuracy.epoch_cache))

        if self.best_model[0] < self.accuracy.epoch_cache.avg:
            self.log.info('beat best ..')
            last_acc = self.best_model[0]
            self.best_model = (self.accuracy.epoch_cache.avg,
                               (self.state_dict())
                               
            )
            self.save_best_model()
            
            if self.config.CONFIG.cuda:
                self.cuda()


        self.test_loss.clear_cache()
        self.accuracy.clear_cache()
        
        for m in self.metrics:
            m.write_to_file()
            
        if self.early_stopping:
            return self.loss_trend()
    
