#!/usr/bin/python
from __future__ import division
from Agent8 import *
from data_reader_general import *
from config import config 
from Layer import GloveMaskCat
import pickle
import numpy as np
import codecs
import copy
import os
from torch import optim
import sys
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

id2label = ["positive", "neutral", "negative"]
#Load concatenation layer and attention model layer
cat_layer = GloveMaskCat(config)
if config.if_gpu:cat_layer = cat_layer.cuda()


def train():
    print(config)
    best_acc = 0
    best_model = None

    # TRAIN_DATA_PATH = "data/restaurant/Restaurants_Train_v2.xml"
    # TEST_DATA_PATH = "data/restaurant/Restaurants_Test_Gold.xml"
    # path_list = [TRAIN_DATA_PATH, TEST_DATA_PATH]
    # #First time, need to preprocess and save the data
    # #Read XML file
    # dr = data_reader(config)
    # dr.read_train_test_data(path_list)
    # print('Data Preprocessed!')



    #Load preprocessed data directly
    dr = data_reader(config)
    train_data = dr.load_data(config.data_path+'Restaurants_Train_v2.xml.pkl')
    test_data = dr.load_data(config.data_path+'Restaurants_Test_Gold.xml.pkl')
    dg_train = data_generator(config, train_data, False)
    dg_test =data_generator(config, test_data, False)

    model = Agent(config)
    if config.if_gpu: model = model.cuda()
#     path = 'rl_model.pt'
#     if os.path.exists(config.model_path+path):
#         print('Loading pretrained model....')
#         file = torch.load(config.model_path+path)
#         model.load_state_dict(file)

#     visualize_samples(dg_train, model)
#     sys.exit()

    
    cat_layer.load_vector()

    #model = Agent(config)

    
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer = create_opt(parameters, config)

    loops = dg_train.data_len
    
    with open(config.log_path+'rl_log.txt', 'w') as f:
        f.write('Experiment starting.....\n')
    for e in np.arange(config.epochs):
        print('Epoch:', e)
        model.train()
        dg_train.reset_samples()
        batch = 64
        iter_num =int(loops/batch)
        for i in range(iter_num):
            optimizer.zero_grad()
            total_loss1 = 0
            total_loss2 = 0
            #Accumulate gradients
            #One sentence each time
            for _ in np.arange(batch):
                sent_vecs, mask_vecs, label_list, sent_lens, texts = next(dg_train.get_ids_samples())
                #Get embeddings for the sentence and the target, no concatenation
                sent_vecs, target_vecs = cat_layer(sent_vecs, mask_vecs)
                if config.if_gpu:
                    sent_vecs = sent_vecs.cuda()
                    mask_vecs = mask_vecs.cuda()
                    label_list = label_list.cuda()
                    sent_lens = sent_lens.cuda()
                #Pass target vectors to the model
                _, actions, action_loss, classify_loss = model(sent_vecs, mask_vecs, label_list, texts)
                total_loss1 += action_loss
                total_loss2 += classify_loss
            total_loss1 = total_loss1/batch
            total_loss2 = total_loss2/batch
            if config.join_train:
                total_loss = total_loss1 + total_loss2*2
                total_loss.backward()
            else:
                total_loss1.backward(retain_graph=True)
                total_loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm, norm_type=2)
            optimizer.step()
            if i %20 == 0:
                print('Action', actions)
                print('Action Loss:', total_loss1)
                print('Classification Loss:', total_loss2)
        
        acc = evaluate_test(dg_test, model)
        with open(config.log_path+'rl_log.txt', 'a') as f:
            f.write('Epoch Num:'+str(e)+'\n')
            f.write('accuracy:'+str(acc))
            f.write('\n')

        if acc > best_acc: 
            best_acc = acc
            #best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), config.model_path+'rl_model.pt')
            torch.save(cat_layer, config.model_path+'rl_model_layer.pt')

def visualize_samples(dr_test, model):
    '''
    Show examples of actions
    '''
    dr_test.reset_samples()
    model.eval()
    all_counter = dr_test.data_len
    correct_count = 0
    while dr_test.index < 30:
        sent_vecs, mask_vecs, label, sent_len, texts = next(dr_test.get_ids_samples())
        sent, target = cat_layer(sent_vecs, mask_vecs)
        if config.if_gpu: 
            sent, target = sent.cuda(), target.cuda()
            label, sent_len = label.cuda(), sent_len.cuda()
        
        pred_label, actions  = model.predict(sent, mask_vecs) 
        if pred_label[0] == label[0]:
            print('*'*20)
            print(texts)
            print('Targets:')
            print(mask_vecs)
            print('Label:', label)
            print('Actions:')
            print(actions)

def evaluate_test(dr_test, model):
    print("Evaluting")
    dr_test.reset_samples()
    model.eval()
    all_counter = dr_test.data_len
    correct_count = 0
    while dr_test.index < dr_test.data_len:
        sent_vecs, mask_vecs, label, sent_len, texts = next(dr_test.get_ids_samples())
        sent, target = cat_layer(sent_vecs, mask_vecs, False)
        if config.if_gpu: 
            sent, target = sent.cuda(), target.cuda()
            label, sent_len = label.cuda(), sent_len.cuda()
        pred_label, _  = model.predict(sent, mask_vecs) 

        correct_count += sum(pred_label==label).item()
    if dr_test.data_len < 1:
        print('Testing Data Error')
    acc = correct_count * 1.0 / dr_test.data_len
    print("Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
    return acc

if __name__ == "__main__":
    train()