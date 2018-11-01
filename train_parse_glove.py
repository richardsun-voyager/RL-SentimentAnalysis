#!/usr/bin/python
from __future__ import division
from model_parse_glove import *
from data_reader_general import *
from parse_path import dependency_path, constituency_path
from configs.config_parse import config
import pickle
from Layer import GloveMaskCat
import numpy as np
import codecs
import copy
import os, sys

def adjust_learning_rate(optimizer, epoch):
    lr = config.lr / (1.5 ** (epoch // config.adjust_every))
    print("Adjust lr to ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_opt(parameters, config):
    if config.opt == "SGD":
        optimizer = optim.SGD(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adam":
        optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adadelta":
        optimizer = optim.Adadelta(parameters, lr=config.lr)
    elif config.opt == "Adagrad":
        optimizer = optim.Adagrad(parameters, lr=config.lr)
    return optimizer

def load_data(data_path, if_utf=False):
    f = open(data_path, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


#id2word = load_data('data/bailin_data/dic.pkl')
id2label = ["positive", "neutral", "negative"]
#Load concatenation layer and attention model layer
cat_layer = GloveMaskCat(config)
dp = dependency_path()
cp = constituency_path()
def convert_mask_index(masks):
    '''
    Find the indice of none zeros values in masks, namely the target indice
    '''
    target_indice = []
    for mask in masks:
        indice = torch.nonzero(mask == 1).squeeze(1).numpy()
        target_indice.append(indice)
    return target_indice

def get_dependency_weight(tokens, targets, max_len):
    '''
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
    return torch.FloatTensor(weights)

def get_context_weight(tokens, targets, max_len):
    weights = np.zeros([len(tokens), max_len])
    for i, token in enumerate(tokens):
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
    return torch.FloatTensor(weights)


def train():
    print(config)
    best_acc = 0
    best_model = None

    # #Load and preprocess raw dataset
    # TRAIN_DATA_PATH = "data/2014/Restaurants_Train_v2.xml"
    # TEST_DATA_PATH = "data/2014/Restaurants_Test_Gold.xml"
    # path_list = [TRAIN_DATA_PATH, TEST_DATA_PATH]
    # #First time, need to preprocess and save the data
    # #Read XML file
    # dr = data_reader(config)
    # dr.read_train_test_data(path_list)
    # print('Data Preprocessed!')
    # dr = data_reader(config)
    # train_data = dr.load_data(config.data_path+'Restaurants_Train_v2.xml.pkl')
    # dr.split_save_data(config.train_path, config.valid_path)



    #Load preprocessed data directly
    dr = data_reader(config)
    train_data = dr.load_data(config.train_path)
    valid_data = dr.load_data(config.valid_path)
    test_data = dr.load_data(config.data_path+'Restaurants_Test_Gold.xml.pkl')
    print('Training Samples:', len(train_data))
    print('Validating Samples:', len(valid_data))
    print('Testing Samples:', len(test_data))

    dg_train = data_generator(config, train_data)
    dg_valid =data_generator(config, valid_data, False)
    dg_test =data_generator(config, test_data, False)

    # dr_valid.load_data(config.valid_path)
    # dr_test = dr_valid
    # dr_test = data_reader(config, False)
    # dr_test.load_data(config.test_path)

    

    cat_layer.load_vector()


    model = depTSA(config)

    # ###Bailin
    if config.if_gpu: model = model.cuda()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # pdb.set_trace()
    optimizer = create_opt(parameters, config)

    with open(config.log_path+'parse_log.txt', 'w') as f:
        f.write('Start Experiment\n')

    loops = int(dg_train.data_len/config.batch_size)
    for e_ in range(config.epoch):
        print("Epoch ", e_ + 1)
        model.train()
        if e_ % config.adjust_every == 0:  
            adjust_learning_rate(optimizer, e_)

        for _ in np.arange(loops):
            model.zero_grad() 
            sent_vecs, mask_vecs, label_list, sent_lens, tokens = next(dg_train.get_ids_samples())
            target_indice = convert_mask_index(mask_vecs)#Get target indice
            max_len = max(sent_lens).item()
           #weights = get_dependency_weight(tokens, target_indice, max_len)#Get weights for each sentence
            weights = get_context_weight(tokens, target_indice, max_len)
            sent_vecs, target_avg = cat_layer(sent_vecs, mask_vecs)#Batch_size*max_len*(2*emb_size)
            if config.if_gpu: 
                sent_vecs, target_avg = sent_vecs.cuda(), target_avg.cuda()
                label_list, sent_lens = label_list.cuda(), sent_lens.cuda()

            cls_loss = model(sent_vecs, weights, label_list, sent_lens)
            l2_loss = 0
            for w in model.parameters():
                if w is not None:
                    l2_loss += w.norm(2)
            print("cls loss {0} regularizrion loss {1}".format(cls_loss.item(), l2_loss.item()))
            cls_loss += l2_loss * 0.005
            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm, norm_type=2)
            optimizer.step()

        acc = evaluate_test(dg_valid, model)
        if acc > best_acc: 
            best_acc = acc
            best_model = copy.deepcopy(model)
            torch.save(best_model, config.model_path+'parse_model.pt')
        with open(config.log_path+'parse_log.txt', 'a') as f:
            f.write('Epoch '+str(e_)+'\n')
            f.write('Validation accuracy:'+str(acc)+'\n')
            if e_ % 1 == 0:
                print('Testing....')
                acc = evaluate_test(dg_test, model)
                f.write('Testing accuracy:'+str(acc)+'\n')

    

def visualize_attention(dr_test, model):
    print("Evaluting")
    dr_test.reset_samples()
    model.eval()
    all_counter = 0
    correct_count = 0
    while dr_test.index < dr_test.data_len:
        sent, mask, label, sent_len, tokens = next(dr_test.get_ids_samples())
        sent, target = cat_layer(sent, mask)
        if config.if_gpu: 
            sent, target = sent.cuda(), target.cuda()
            label, sent_len = label.cuda(), sent_len.cuda()
        pred_label, attentions  = model.predict(sent, target, sent_len) 
        print(tokens[0])
        print(mask[0])
        print(attentions[0])
        break


def evaluate_test(dr_test, model):
    print("Evaluting")
    dr_test.reset_samples()
    model.eval()
    all_counter = 0
    correct_count = 0
    while dr_test.index < dr_test.data_len:
        sent_vecs, mask_vecs, label_list, sent_lens, tokens = next(dr_test.get_ids_samples())
        target_indice = convert_mask_index(mask_vecs)#Get target indice
        max_len = max(sent_lens).item()
        weights = get_context_weight(tokens, target_indice, max_len)#Get weights for each sentence
        sent_vecs, target_avg = cat_layer(sent_vecs, mask_vecs)#Batch_size*max_len*(2*emb_size)
        if config.if_gpu: 
            sent_vecs, target_avg = sent_vecs.cuda(), target_avg.cuda()
            label_list, sent_lens = label_list.cuda(), sent_lens.cuda()

        pred_label, _  = model.predict(sent_vecs, weights, sent_lens) 

        correct_count += sum(pred_label==label_list).item()
    if dr_test.data_len < 1:
        print('Testing Data Error')
    acc = correct_count * 1.0 / dr_test.data_len
    print("Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
    return acc

if __name__ == "__main__":
    train()