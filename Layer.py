from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import pdb
import pickle
import numpy as np
import math
from torch.nn import utils as nn_utils
from util import *


def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])


    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    pos_emb = torch.from_numpy(position_enc).type(torch.FloatTensor)
    return pos_emb

# input layer for 14
class SimpleCat(nn.Module):
    def __init__(self, config):
        '''
        Concatenate word embeddings and target embeddings
        '''
        super(SimpleCat, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(config.embed_num, config.embed_dim)
        self.mask_embed = nn.Embedding(2, config.mask_dim)

        self.dropout = nn.Dropout(config.rnn_dropout)

    # input are tensors
    def forward(self, sent, mask, is_elmo=True):
        '''
        Args:
        sent: tensor, shape(batch_size, max_len, emb_dim)
        mask: tensor, shape(batch_size, max_len)
        '''
        #Modified by Richard Sun
        #Use ELmo embedding, note we need padding

        #Use GloVe embedding
        if self.config.if_gpu:  
            sent, mask = sent.cuda(), mask.cuda()
        # to embeddings
        sent_vec = sent # batch_siz*sent_len * dim
        if is_elmo:
            sent_vec = sent # batch_siz*sent_len * dim
        else:
            sent_vec = self.word_embed(sent)# batch_siz*sent_len * dim
            
        mask_vec = self.mask_embed(mask) # batch_size*max_len* dim
        #print(mask_vec.size())

        #sent_vec = self.dropout(sent_vec)
        #Concatenation
        sent_vec = torch.cat([sent_vec, mask_vec], 2)

        # for test
        return sent_vec

    def load_vector(self):
        with open(self.config.embed_path, 'rb') as f:
            vectors = pickle.load(f)
            print("Loaded from {} with shape {}".format(self.config.embed_path, vectors.shape))
            #self.word_embed.weight = nn.Parameter(torch.FloatTensor(vectors))
            self.word_embed.weight.data.copy_(torch.from_numpy(vectors))
            # self.word_embed.weight.requires_grad = False
    
    def reset_binary(self):
        self.mask_embed.weight.data[0].zero_()

        
    # input layer for 14
class WordPosMaskCat(nn.Module):
    def __init__(self, config):
        super(WordPosMaskCat, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(config.embed_num, config.embed_dim)
        self.mask_embed = nn.Embedding(2, config.mask_dim)
        
        n_position = 100
        self.position_enc = nn.Embedding(n_position, config.mask_dim, padding_idx=0)
        self.position_enc.weight.data = position_encoding_init(n_position, config.mask_dim)
        self.load_vector()
        self.dropout = nn.Dropout(config.rnn_dropout)

    # input are tensors
    def forward(self, sents, masks, positions, is_avg=True):
        '''
        Args:
        sent: tensor, shape(batch_size, max_len)
        mask: tensor, shape(batch_size, max_len)
        positions: tensor, shape(batch_size, max_len)
        '''
        #Modified by Richard Sun
        #Use ELmo embedding, note we need padding

        #Use GloVe embedding
          
        # to embeddings
        sent_vec = self.word_embed(sents) # batch_siz*sent_len * dim
        pos_vec = self.position_enc(positions)
        #mask_emb = self.mask_embed(masks)
        #Concatenate each word embedding with target word embeddings' average
        sent_vec_cat = torch.cat([sent_vec, pos_vec], 2)
        if self.config.if_gpu:
            sent_vec_cat = sent_vec_cat.cuda()

        return sent_vec_cat

    def load_vector(self):
        with open(self.config.embed_path, 'rb') as f:
            #vectors = pickle.load(f, encoding='bytes')
            vectors = pickle.load(f)
            print("Loaded from {} with shape {}".format(self.config.embed_path, vectors.shape))
            #self.word_embed.weight = nn.Parameter(torch.FloatTensor(vectors))
            self.word_embed.weight.data.copy_(torch.from_numpy(vectors))
            self.word_embed.weight.requires_grad = self.config.if_update_embed
    
    def reset_binary(self):
        self.mask_embed.weight.data[0].zero_()


    # input layer for 14
class GloveMaskCat(nn.Module):
    def __init__(self, config):
        super(GloveMaskCat, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(config.embed_num, config.embed_dim)
        self.mask_embed = nn.Embedding(2, config.mask_dim)
        
        n_position = 100
        self.position_enc = nn.Embedding(n_position, config.embed_dim, padding_idx=0)
        self.position_enc.weight.data = position_encoding_init(n_position, config.embed_dim)
        #self.load_vector()
        self.dropout = nn.Dropout(config.rnn_dropout)

    # input are tensors
    def forward(self, sents, masks, is_avg=True):
        '''
        Args:
        sent: tensor, shape(batch_size, max_len)
        mask: tensor, shape(batch_size, max_len)
        '''
        #Modified by Richard Sun
        #Use ELmo embedding, note we need padding

        #Use GloVe embedding
        if self.config.if_gpu:  
            sents, masks = sents.cuda(), masks.cuda()
        # to embeddings
        sent_vec = self.word_embed(sents) # batch_siz*sent_len * dim
        #Concatenate each word embedding with target word embeddings' average
        batch_size, max_len = sents.size()
        #Get the index of target
        #target_index = [torch.nonzero(mask).squeeze(1) for mask in masks]
        #Repeat the mask
        masks = masks.type_as(sent_vec)
        masks = masks.expand(self.config.embed_dim, batch_size, max_len)
        masks = masks.transpose(0, 1).transpose(1, 2)#The same size as sentence vector
        target_emb = sent_vec * masks
        target_emb_avg = torch.sum(target_emb, 1)/torch.sum(masks, 1)#Batch_size*embedding
        #Expand dimension for concatenation
        target_emb_avg_exp = target_emb_avg.expand(max_len, batch_size, self.config.embed_dim)
        target_emb_avg_exp = target_emb_avg_exp.transpose(0, 1)#Batch_size*max_len*embedding
        if is_avg:
            target = target_emb_avg
        else:
            target = target_emb

        #sent_vec = self.dropout(sent_vec)
        #Concatenation
        #sent_target_concat = torch.cat([sent_vec, target_emb_avg_exp], 2)

        # for test
        return sent_vec, target

    def load_vector(self):
        with open(self.config.embed_path, 'rb') as f:
            #vectors = pickle.load(f, encoding='bytes')
            vectors = pickle.load(f)
            print("Loaded from {} with shape {}".format(self.config.embed_path, vectors.shape))
            #self.word_embed.weight = nn.Parameter(torch.FloatTensor(vectors))
            self.word_embed.weight.data.copy_(torch.from_numpy(vectors))
            self.word_embed.weight.requires_grad = self.config.if_update_embed
    
    def reset_binary(self):
        self.mask_embed.weight.data[0].zero_()


# input layer for 14
class ContextTargetCat(nn.Module):
    def __init__(self, config):
        super(ContextTargetCat, self).__init__()
        '''
        This class is to concatenate the context and target embeddings
        '''
        self.config = config
        self.word_embed = nn.Embedding(config.embed_num, config.embed_dim)
        self.mask_embed = nn.Embedding(2, config.mask_dim)
        

        self.dropout = nn.Dropout(config.rnn_dropout)

    # input are tensors
    def forward(self, sent, mask, is_concat=True):
        '''
        Args:
        sent: tensor, shape(batch_size, max_len, dim) elmo
        mask: tensor, shape(batch_size, max_len)
        '''
        #Modified by Richard Sun
        #Use ELmo embedding, note we need padding


        if self.config.if_gpu:  
            sent, mask = sent.cuda(), mask.cuda()
        # # to embeddings
        
        batch_size, max_len, _ = sent.size()
        sent_vec = sent
        if is_concat:
            ## concatenate each word embedding with mask embedding
            mask_emb = self.mask_embed(mask)
            sent_target_concat = torch.cat([sent_vec, mask_emb], 2)
            sent_target = sent_target_concat
        else:#Add each word embedding with target word embeddings' average
            #Repeat the mask
            mask = mask.type_as(sent_vec)
            mask = mask.expand(self.config.embed_dim, batch_size, max_len)
            mask = mask.transpose(0, 1).transpose(1, 2)#The same size as sentence vector
            target_emb = sent_vec * mask
            target_emb_avg = torch.sum(target_emb, 1)/torch.sum(mask, 1)#Batch_size*embedding
            #Expand dimension for concatenation
            target_emb_avg_exp = target_emb_avg.expand(max_len, batch_size, self.config.embed_dim)
            target_emb_avg_exp = target_emb_avg_exp.transpose(0, 1)#Batch_size*max_len*embedding
            sent_target = (sent_vec + target_emb_avg_exp)/2


        
        #sent_vec = self.dropout(sent_vec)

        # for test
        return sent_target

    def load_vector(self):
        with open(self.config.embed_path, 'rb') as f:
            #vectors = pickle.load(f, encoding='bytes')
            vectors = pickle.load(f)
            print("Loaded from {} with shape {}".format(self.config.embed_path, vectors.shape))
            #self.word_embed.weight = nn.Parameter(torch.FloatTensor(vectors))
            self.word_embed.weight.data.copy_(torch.from_numpy(vectors))
            self.word_embed.weight.requires_grad = False
    
    def reset_binary(self):
        self.mask_embed.weight.data[0].zero_()
        
        

