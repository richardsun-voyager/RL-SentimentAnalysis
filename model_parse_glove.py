from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb
import pickle
#from parse_path import constituency_path
import math
from util import *
from torch.nn import utils as nn_utils
import torch.nn.init as init
def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)

class MLSTM(nn.Module):
    def __init__(self, config):
        super(MLSTM, self).__init__()
        self.config = config
        #The concatenated word embedding and target embedding as input
        self.rnn = nn.LSTM(config.embed_dim , int(config.l_hidden_size / 2), batch_first=True, num_layers = int(config.l_num_layers / 2),
            bidirectional=True, dropout=config.l_dropout)
        init_ortho(self.rnn)

    # batch_size * sent_l * dim
    def forward(self, feats, seq_lengths=None):
        '''
        Args:
        feats: batch_size, max_len, emb_dim
        seq_lengths: batch_size
        '''
        #FIXIT: doesn't have batch
        #Sort the lengths
        # seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        # feats = feats[perm_idx]
        #feats = feats.unsqueeze(0)
        pack = nn_utils.rnn.pack_padded_sequence(feats, 
                                                 seq_lengths, batch_first=True)
        
        
        #assert self.batch_size == batch_size
        lstm_out, _ = self.rnn(pack)
        #lstm_out, (hid_states, cell_states) = self.rnn(feats)

        #Unpack the tensor, get the output for varied-size sentences
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        #FIXIT: for batch
        #lstm_out = lstm_out.squeeze(0)
        # batch * sent_l * 2 * hidden_states 
        return unpacked

# consits of three components
class depTSA(nn.Module):
    def __init__(self, config):
        super(depTSA, self).__init__()
        self.config = config

        self.lstm = MLSTM(config)
        self.target2vec = nn.Linear(config.embed_dim, config.l_hidden_size)
        self.vec2label = nn.Linear(config.embed_dim, 3)
        self.concatvec_linear = nn.Linear(2*config.l_hidden_size, 1)

        self.cri = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(config.dropout)
        #Modified by Richard Sun
        #If we use Elmo, don't load GloVec
        #self.cat_layer.load_vector()

    def compute_score(self, sent, weights, lens):
        '''
        inputs are list of list for the convenince of top CRF
        Args:
        sent: a list of sentences， batch_size*max_len*(2*emb_dim)
        weights: batch_size*max_len
        label: a list labels
        '''

        #Get the target embedding
        #batch_size, sent_len, dim = sent.size()
        weights = weights.unsqueeze(1)#batch_size*1*max_len

        sents_vec = torch.bmm(weights, sent).squeeze(1)#Batch_size*hidden_dim
        
        #Dropout
        if self.training:
            sents_vec = self.dropout(sents_vec)

        output = self.vec2label(sents_vec)#Bach_size*label_size

        scores = F.log_softmax(output, dim=1)#Batch_size*label_size
        return scores, weights

   
    def forward(self, sent, weights, label, lens):
        #Sent emb_dim + 50
        
        sent = F.dropout(sent, p=0.2, training=self.training)
        scores, _ = self.compute_score(sent, weights, lens)
        loss = nn.NLLLoss()
        #cls_loss = -1 * torch.log(scores[label])
        cls_loss = loss(scores, label)

        #print('Transition', pena)
        return cls_loss 

    def predict(self, sent, weights, sent_len):
        #sent = self.cat_layer(sent, mask)
        scores, attentions = self.compute_score(sent, weights, sent_len)
        _, pred_label = scores.max(1)#Find the max label in the 2nd dimension
        
        #Modified by Richard Sun
        return pred_label, attentions

