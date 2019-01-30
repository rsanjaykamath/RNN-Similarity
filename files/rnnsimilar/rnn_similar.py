import logging

import torch
import torch.nn as nn

from . import layers

logger = logging.getLogger(__name__)


class RnnSimilar(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(RnnSimilar, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size : word emb + question emb + manual features
        doc_input_size = args.embedding_dim + args.num_features

        if args.use_qemb:
            doc_input_size += args.embedding_dim

        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )
        self.question_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.question_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers

        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_q_attn = layers.LinearSeqAttn(question_hidden_size)

            self.self_d_attn = layers.LinearSeqAttn(doc_hidden_size)

        self.binary_op = layers.Binary_Softmax(question_hidden_size * 2, doc_hidden_size, normalize=normalize)


    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):


        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)


        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)

        drnn_input = [x1_emb]

        if self.args.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input.append(x2_weighted_emb)

        if self.args.num_features > 0:
            drnn_input.append(x1_f)

        doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask)
        question_hiddens = self.question_rnn(x2_emb, x2_mask)

        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
            d_merge_weights = layers.uniform_weights(doc_hiddens, x1_mask)

        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_q_attn(question_hiddens, x2_mask)
            d_merge_weights = self.self_d_attn(doc_hiddens, x1_mask)

        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)
        doc_hidden = layers.weighted_avg(doc_hiddens, d_merge_weights)
        pred_binary = self.binary_op(question_hidden, doc_hidden)

        return pred_binary