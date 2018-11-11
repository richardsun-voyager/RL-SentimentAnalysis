from config_imdb import config 
from data_reader_imdb import dataHelper
from simple_classification_models.cnn_classifier import CNNClassifier
from simple_classification_models.rnn_classifier import RNNClassifier
from torch import nn, optim
import numpy as np
import torch

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

def train():
    dh = dataHelper(config)
    dh.read_csv_data('data/IMDB/train.csv')
    loss = nn.NLLLoss()
    
    model = RNNClassifier(config)
    
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = create_opt(parameters, config)
    
    loops = int(dh.data_len/config.batch_size)
    for i in np.arange(10):
        for j in np.arange(loops):
            optimizer.zero_grad() 
            sents, labels, lengths = dh.get_ids_samples()
            preds = model(sents, lengths)
            cost = loss(preds, labels)
            cost.backward()
            optimizer.step()
            if j%20 == 0:
                print(cost.item())
                
      
            
        
    

if __name__ == "__main__":
    train()