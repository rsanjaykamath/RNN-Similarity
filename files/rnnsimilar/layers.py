import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
from torch.autograd import Variable


logger = logging.getLogger(__name__)

class StackedBRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):

        if x_mask.data.sum() == 0:
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            output = self._forward_padded(x, x_mask)
        else:
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        x = x.transpose(0, 1)

        outputs = [x]

        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            rnn_output, rnn_hidden = self.rnns[i](rnn_input)
            outputs.append(rnn_output)

        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        output = output.transpose(0, 1)

        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):

        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        x = x.index_select(0, idx_sort)

        x = x.transpose(0, 1)

        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        outputs = [rnn_input]
        hiddens = []
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)

            outs = self.rnns[i](rnn_input)
            outputs.append(outs[0])

        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

class SeqAttnMatch(nn.Module):

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)

        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        scores = x_proj.bmm(y_proj.transpose(2, 1))

        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=-1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        matched_seq = alpha.bmm(y)
        return matched_seq


class Binary_Softmax(nn.Module):

    def __init__(self, x_size, y_size, normalize=True):
        super(Binary_Softmax, self).__init__()

        self.normalize = normalize
        self.linear = nn.Linear(x_size, 1)

    def forward(self, x, y):

        inp = [x]
        inp.append(y)
        representation = torch.cat(inp, -1)
        output = self.linear(representation)

        return F.sigmoid(output)



class LinearSeqAttn(nn.Module):
    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        return alpha


def uniform_weights(x, x_mask):
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    return weights.unsqueeze(1).bmm(x).squeeze(1)

