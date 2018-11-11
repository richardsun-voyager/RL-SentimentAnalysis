import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn import utils as nn_utils

def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)
            
class RNNClassifier(nn.Module):
    def __init__(self, config, output_dim=3):
        super(RNNClassifier, self).__init__()
        self.config = config
        #The concatenated word embedding and target embedding as input
        self.rnn = DynamicRNN(config)
        self.fc = nn.Linear(config.l_hidden_size, output_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.activate = nn.LogSoftmax(dim=1)

    # batch_size * sent_l * dim
    def forward(self, feats, seq_lengths=None):
        '''
        Args:
        feats: batch_size, max_len, emb_dim
        seq_lengths: batch_size
        '''
        unpacked, h = self.rnn(feats, seq_lengths)
        #h: batch_size, 2*config.hidden_size
        h = self.dropout(h)
        output = self.fc(h)
        output = self.activate(output)
        #lstm_out = lstm_out.squeeze(0)
        # batch * sent_l * 2 * hidden_states 
        return output
            
class DynamicRNN(nn.Module):
    def __init__(self, config):
        super(DynamicRNN, self).__init__()
        self.config = config
        #The concatenated word embedding and target embedding as input
        self.rnn = nn.LSTM(config.embed_dim , int(config.l_hidden_size / 2), batch_first=True, num_layers = int(config.l_num_layers / 2), bidirectional=True, dropout=config.rnn_dropout)
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
        lstm_out, (h, c) = self.rnn(pack)
        #lstm_out, (hid_states, cell_states) = self.rnn(feats)

        #Unpack the tensor, get the output for varied-size sentences
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        #FIXIT: for batch
        h = h.transpose(0, 1).contiguous().view(len(feats), -1)
        #lstm_out = lstm_out.squeeze(0)
        # batch * sent_l * 2 * hidden_states 
        return unpacked, h