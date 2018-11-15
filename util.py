from collections import namedtuple
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb

from parse_path import constituency_path, dependency_path
dp = dependency_path()
cp = constituency_path()
def convert_mask_index(masks):
    '''
    Find the indice of none zeros values in masks, namely the target indice
    '''
    target_indice = []
    for mask in masks:
        indice = torch.nonzero(mask == 1).squeeze(1).cpu().numpy()
        target_indice.append(indice)
    return target_indice

def get_dependency_weight(tokens, targets, max_len):
    '''
    Dependency weight
    tokens: texts
    max_len: max length of texts
    '''
    weights = np.zeros([len(tokens), max_len])
    for i, token in enumerate(tokens):
        try:
            graph = dp.build_graph(token)
            mat = dp.compute_node_distance(graph, max_len)
        except:
            print('Error!!!!!!!!!!!!!!!!!!')
            print(text)

        try:
            max_w, _, _ = dp.compute_soft_targets_weights(mat, targets[i])
            weights[i, :len(max_w)] = max_w
        except:
            print('text process error')
            print(text, targets[i])
            break
    return weights

def get_context_weight(texts, targets, max_len):
    '''
    Constituency weight
    '''
    weights = np.zeros([len(texts), max_len])
    for i, token in enumerate(texts):
        #print('Original word num')
        #print(len(token))
        #text = ' '.join(token)#Connect them into a string
        #stanford nlp cannot identify the abbreviations ending with '.' in the sentences

        try:
            max_w, min_w, a_v = cp.proceed(token, targets[i])
            weights[i, :len(max_w)] = max_w
        except Exception as e:
            print(e)
            print(token, targets[i])
    return weights

def get_target_emb(sent_vec, masks, is_average=True):
    '''
    '''
    batch_size, max_len, embed_dim = sent_vec.size()
    masks = masks.type_as(sent_vec)
    masks = masks.expand(embed_dim, batch_size, max_len)
    masks = masks.transpose(0, 1).transpose(1, 2)#The same size as sentence vector
    target_emb = sent_vec * masks
    if is_average:
        target_emb_avg = torch.sum(target_emb, 1)/torch.sum(masks, 1)#Batch_size*embedding
        return target_emb_avg
    return target_emb

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # vec is only 1d vec
    # return the argmax as a python int
    _, idx = torch.max(vec, 0)
    return to_scalar(idx)

# the input is 2d dim tensor
# output 1d tensor
def argmax_m(mat):
    ret_v, ret_ind = [], []
    m, n = mat.size()
    for i in range(m):
        ind_ = argmax(mat[i])
        ret_ind.append(ind_)
        ret_v.append(mat[i][ind_])
    if type(ret_v[0]) == Variable or type(ret_v[0]) == torch.Tensor:
        return ret_ind, torch.stack(ret_v)
    else:
        return ret_ind, torch.Tensor(ret_v)

# Compute log sum exp in a numerically stable way for the forward algorithm
# vec is n * n, norm in row
def log_sum_exp_m(mat):
    row, column = mat.size()
    ret_l = []
    for i in range(row):
        vec = mat[i]
        max_score = vec[argmax(vec)]
        max_score_broadcast = max_score.view( -1).expand(1, vec.size()[0])
        ret_l.append( max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast))))
    return torch.stack(ret_l)

def log_sum_exp(vec_list):
    tmp_mat = torch.stack(vec_list, 0)
    m,n = tmp_mat.size()
    # value may be nan because of gradient explosion
    try:
        max_score = torch.max(tmp_mat)
    except:
        pdb.set_trace()
    max_expand = max_score.expand(m, n)
    max_ex_v = max_score.expand(1, n)
    # sum along dim 0
    ret_val = max_ex_v + torch.log(torch.sum(torch.exp(tmp_mat - max_expand), 0))
    return ret_val

# vec1 and vec2 both 1d tensor
# return 1d tensor
def add_broad(vec1, vec2):
    s_ = vec1.size()[0]
    vec1 = vec1.expand(3, s_).transpose(0,1)
    vec2 = vec2.expand(s_, 3)
    new_vec = vec1 + vec2
    return new_vec.view(-1)

# transform a list to 1d vec
def to_1d(vec_list):
    ret_v = vec_list[0].clone()
    v_l = len(vec_list)
    for i in range(1, v_l):
        ret_v = add_broad(ret_v, vec_list[i])
    return ret_v

def to_ind(num, logit):
    ret_l = []
    for i in reversed(range(logit)):
        tmp = num / 3 ** i
        num = num - tmp * 3 ** i
        ret_l.append(tmp)
    return list(reversed(ret_l))

def create_empty_var(if_gpu):
    if if_gpu:
        loss = Variable(torch.Tensor([0]).cuda())
    else:
        loss = Variable(torch.Tensor([0])) 
    return loss