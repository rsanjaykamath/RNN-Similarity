import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import numpy as np
import logging
import copy
import torch.nn as nn

from torch.autograd import Variable
from .config import override_model_args
from .rnn_similar import RnnSimilar
from .data import Dictionary

logger = logging.getLogger(__name__)


class RnnSimilarModel(object):

    def __init__(self, args, word_dict, feature_dict,
                 state_dict=None, normalize=True):
        self.args = args
        self.word_dict = word_dict
        self.args.vocab_size = len(word_dict)
        self.feature_dict = feature_dict
        self.args.num_features = len(feature_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        if args.model_type == 'rnn':
            self.network = RnnSimilar(args, normalize)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)


        if state_dict:
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)


    def expand_dictionary(self, words):
        to_add = {self.word_dict.normalize(w) for w in words
                  if w not in self.word_dict}

        if len(to_add) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add))
            for w in to_add:
                self.word_dict.add(w)
            self.args.vocab_size = len(self.word_dict)
            logger.info('New vocab size: %d' % len(self.word_dict))

            old_embedding = self.network.embedding.weight.data
            self.network.embedding = torch.nn.Embedding(self.args.vocab_size,
                                                        self.args.embedding_dim,
                                                        padding_idx=0)
            new_embedding = self.network.embedding.weight.data
            new_embedding[:old_embedding.size(0)] = old_embedding

        return to_add



    def load_embeddings(self, words, embedding_file):
        words = {w for w in words if w in self.word_dict}

        x3 = np.random.uniform(-0.5, 0.5, 300)
        word3 = list(x3)
        word3 = [float("{0:.5f}".format(x)) for x in word3]

        x2 = np.random.uniform(-0.5, 0.5, 300)
        word2 = list(x2)
        word2 = [float("{0:.5f}".format(x)) for x in word2]

        x1 = np.random.uniform(-0.5, 0.5, 300)
        word1 = list(x1)
        word1 = [float("{0:.5f}".format(x)) for x in word1]

        x4 = np.random.uniform(-0.5, 0.5, 300)
        word4 = list(x4)
        word4 = [float("{0:.5f}".format(x)) for x in word4]

        x5 = np.random.uniform(-0.5, 0.5, 300)
        word5 = list(x5)
        word5 = [float("{0:.5f}".format(x)) for x in word5]

        x6 = np.random.uniform(-0.5, 0.5, 300)
        word6 = list(x6)
        word6 = [float("{0:.5f}".format(x)) for x in word6]

        x7 = np.random.uniform(-0.5, 0.5, 300)
        word7 = list(x7)
        word7 = [float("{0:.5f}".format(x)) for x in word7]

        x8 = np.random.uniform(-0.5, 0.5, 300)
        word8 = list(x8)
        word8 = [float("{0:.5f}".format(x)) for x in word8]

        x9 = np.random.uniform(-0.5, 0.5, 300)
        word9 = list(x9)
        word9 = [float("{0:.5f}".format(x)) for x in word9]

        x10 = np.random.uniform(-0.5, 0.5, 300)
        word10 = list(x10)
        word10 = [float("{0:.5f}".format(x)) for x in word10]

        x11 = np.random.uniform(-0.5, 0.5, 300)
        word11 = list(x11)
        word11 = [float("{0:.5f}".format(x)) for x in word11]

        x12 = np.random.uniform(-0.5, 0.5, 300)
        word12 = list(x12)
        word12 = [float("{0:.5f}".format(x)) for x in word12]

        x13 = np.random.uniform(-0.5, 0.5, 300)
        word13 = list(x13)
        word13 = [float("{0:.5f}".format(x)) for x in word13]

        x14 = np.random.uniform(-0.5, 0.5, 300)
        word14 = list(x14)
        word14 = [float("{0:.5f}".format(x)) for x in word14]

        x15 = np.random.uniform(-0.5, 0.5, 300)
        word15 = list(x15)
        word15 = [float("{0:.5f}".format(x)) for x in word15]


        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        vec_counts = {}

        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert (len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

            v1 = torch.Tensor([i for i in word1])
            vec_counts["wikipage_hum"] = 1
            embedding[self.word_dict["wikipage_hum"]].add_(v1)

            v2 = torch.Tensor([i for i in word2])
            vec_counts["wikipage_loc"] = 1
            embedding[self.word_dict["wikipage_loc"]].add_(v2)

            v3 = torch.Tensor([i for i in word3])
            vec_counts["wikipage_enty"] = 1
            embedding[self.word_dict["wikipage_enty"]].add_(v3)

            v4 = torch.Tensor([i for i in word4])
            vec_counts["wikipage_num"] = 1
            embedding[self.word_dict["wikipage_num"]].add_(v4)

            v5 = torch.Tensor([i for i in word5])
            vec_counts["wikipage_desc"] = 1
            embedding[self.word_dict["wikipage_desc"]].add_(v5)

            v6 = torch.Tensor([i for i in word6])
            vec_counts["wikipage_abbr"] = 1
            embedding[self.word_dict["wikipage_abbr"]].add_(v6)

            v7 = torch.Tensor([i for i in word7])
            vec_counts["max_entity_left"] = 1
            embedding[self.word_dict["max_entity_left"]].add_(v7)

            v8 = torch.Tensor([i for i in word8])
            vec_counts["entity_left"] = 1
            embedding[self.word_dict["entity_left"]].add_(v8)

            v9 = torch.Tensor([i for i in word9])
            vec_counts["wikipage_entity"] = 1
            embedding[self.word_dict["wikipage_entity"]].add_(v9)

            v10 = torch.Tensor([i for i in word10])
            vec_counts["max_entity_hum"] = 1
            embedding[self.word_dict["max_entity_hum"]].add_(v10)

            v11 = torch.Tensor([i for i in word11])
            vec_counts["max_entity_loc"] = 1
            embedding[self.word_dict["max_entity_loc"]].add_(v11)

            v12 = torch.Tensor([i for i in word12])
            vec_counts["max_entity_enty"] = 1
            embedding[self.word_dict["max_entity_enty"]].add_(v12)

            v13 = torch.Tensor([i for i in word13])
            vec_counts["max_entity_num"] = 1
            embedding[self.word_dict["max_entity_num"]].add_(v13)

            v14 = torch.Tensor([i for i in word14])
            vec_counts["max_entity_desc"] = 1
            embedding[self.word_dict["max_entity_desc"]].add_(v14)

            v15 = torch.Tensor([i for i in word15])
            vec_counts["max_entity_abbr"] = 1
            embedding[self.word_dict["max_entity_abbr"]].add_(v15)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))



    def tune_embeddings(self, words):
        words = {w.lower() for w in words if w.lower() in self.word_dict}

        if len(words) == 0:
            logger.warning('Tried to tune embeddings, but no words given!')
            return

        if len(words) == len(self.word_dict):
            logger.warning('Tuning ALL embeddings in dictionary')
            return

        embedding = self.network.embedding.weight.data
        for idx, swap_word in enumerate(words, self.word_dict.START):
            curr_word = self.word_dict[idx]
            curr_emb = embedding[idx].clone()
            old_idx = self.word_dict[swap_word]

            embedding[idx].copy_(embedding[old_idx])
            embedding[old_idx].copy_(curr_emb)
            self.word_dict[swap_word] = idx
            self.word_dict[idx] = swap_word
            self.word_dict[curr_word] = old_idx
            self.word_dict[old_idx] = curr_word

        self.network.register_buffer(
            'fixed_embedding', embedding[idx + 1:].clone()
        )

    def init_optimizer(self, state_dict=None):
        if self.args.fix_embeddings:
            for p in self.network.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)


    def update(self, ex):
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        self.network.train()

        if self.use_cuda:

            inputs = [e if e is None else Variable(e.cuda(async=True))
                      for e in ex[:5]]
            target_s = Variable(ex[5].cuda(async=True))

        else:
            inputs = [e if e is None else Variable(e) for e in ex[:5]]
            target_s = Variable(ex[5])


        score_s = self.network(*inputs)

        criterion = nn.MSELoss()
        loss = criterion(score_s, target_s)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.args.grad_clipping)

        self.optimizer.step()
        self.updates += 1

        self.reset_parameters()

        return loss.data[0], ex[0].size(0)

    def reset_parameters(self):
        if self.args.tune_partial > 0:
            if self.parallel:
                embedding = self.network.module.embedding.weight.data
                fixed_embedding = self.network.module.fixed_embedding
            else:
                embedding = self.network.embedding.weight.data
                fixed_embedding = self.network.fixed_embedding

            offset = embedding.size(0) - fixed_embedding.size(0)
            if offset >= 0:
                embedding[offset:] = fixed_embedding


    def predict(self, ex, candidates=None, top_n=5, async_pool=None):

        self.network.eval()

        if self.use_cuda:
            inputs = [e if e is None else
                      Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:5]]
        else:
            inputs = [e if e is None else Variable(e, volatile=True)
                      for e in ex[:5]]
        score_s = self.network(*inputs)

        score_s = score_s.data.cpu()

        return score_s

    def save(self, filename):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        params = {
            'state_dict': network.state_dict(),
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']

        if new_args:
            args = override_model_args(args, new_args)
        return ReDocReader(args, word_dict, feature_dict, state_dict, normalize)

    @staticmethod
    def load_checkpoint(filename, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = ReDocReader(args, word_dict, feature_dict, state_dict, normalize)
        model.init_optimizer(optimizer)
        return model, epoch


    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
