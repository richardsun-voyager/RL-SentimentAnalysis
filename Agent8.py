import numpy as np
import pickle
import torch
import os
from torch.nn import utils as nn_utils
#from config import config
import torch.nn as nn
import torch.nn.functional as F
from util import *
import torch.nn.init as init
from torch.distributions import Categorical
def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)

class Env(nn.Module):
    def __init__(self, config):
        '''
        Construct an virtual environment that can generate, update state and rewards to the agent
        '''
        super(Env, self).__init__()
        self.config = config
        
        #Current state
        self.observation = {}
        
        #output states of RNNs
        self.cell_state = None
        self.hidden_state = None
        
        #biRNN outputs
        self.contexts = None
        
        #Parsing weights for each word
        self.weights = None
    
        #Encode the entire sentence
        self.birnn = nn.LSTM(config.embed_dim, int(config.hidden_dim/2), batch_first=True, num_layers= config.layer_num, bidirectional=True, dropout=config.rnn_dropout)
        #Compress the sentence
        self.rnn = nn.LSTM(config.embed_dim, config.hidden_dim, batch_first=True, num_layers= config.layer_num, bidirectional=False, dropout=config.rnn_dropout)
        
        init_ortho(self.rnn)
        init_ortho(self.birnn)
        self.load_params()
        
    def load_params(self, file='data/models/pretrained_lstm_params.pkl'):
        '''
        Initialize BiLSTM with pre-trained parameters
        '''
        if os.path.exists(file):
            with open(file, 'rb') as f:
                weight_hh_l0, weight_hh_l0_reverse, weight_ih_l0, weight_ih_l0_reverse = pickle.load(f)
            self.birnn.weight_hh_l0.data.copy_(weight_hh_l0.data)
            self.birnn.weight_hh_l0_reverse.data.copy_(weight_hh_l0_reverse.data)
            self.birnn.weight_ih_l0.data.copy_(weight_ih_l0)
            self.birnn.weight_ih_l0_reverse.data.copy_(weight_ih_l0_reverse.data)
            print('BiLSTM parameters Initialized!')
        else:
            print('Pretrained file not found')
        
    def init_env(self, sents, masks, texts=None):
        '''
        Initialize the enviroment
        Args:
        sents: sequence of word idx
        texts: origin texts
        '''
        self.init_hidden()
        self.get_sent_context_state(sents)
        max_len = sents.size(1)
        if texts:
            self.get_parsing_weights(texts, masks, max_len)
    
    def step(self, action):
        observation = None
        reward = None
        done = None
        return observation, reward, done
        
    def get_parsing_weights(self, texts, masks, max_len):
        '''
        Get sentence's parsing weights according to the target
        Args:
        sents: [batch_size, len], texts
        masks: [batch_size, len], binary
        '''
        ##########Get parsing weights for each sentence given a target
        target_indice = convert_mask_index(masks)#Get target indice
        weights = get_context_weight(texts, target_indice, max_len)
        #Not differentiable
        self.weights = weights
    
    def init_hidden(self):
        '''
        Initialize hidden state, C, H
        '''
        size = int(self.config.hidden_dim)
        h = torch.zeros(1, 1, size)
        c = torch.zeros(1, 1, size)
        if self.config.if_gpu:
            h, c = h.cuda(), c.cuda()
        self.hidden_state, self.cell_state = h, c
        
    def generate_reward(self, action, pos):
        '''
        Give reward for an action, the reward must be reasonable for selecting right words
        '''
        if action == 1:
            parse_reward = self.weights[0][pos]
            if parse_reward > 0.2:
                parse_reward = 1
            del_reward = 0
        else:
            parse_reward = 0
            del_reward = 0.5
        #print(parse_reward)
        total_reward = del_reward+parse_reward
        return total_reward
    
    def generate_critic_action(self, pos):
        if self.weights[0][pos] > 0.2:
            action = 1
            reward = self.weights[0][pos]*2
        else:
            action = 0
            reward = 0.5
        return action, reward

        
    def get_sent_context_state(self, sents):
        '''
        Get the hidden state for the original sentence
        Args:
        texts: a tensor of texts, batch_size, max_len, embed_dim
        '''
        # #1, max_len, hidden_dim
        contexts, _ = self.birnn(sents)
        #1, 1, hidden_dim
        self.contexts = contexts
        
        
    
    #Note the state should be non-differentiable
    def set_current_state(self, pos, current_word, current_pos, target, target_pos):
        '''
        Compute current state, just concatenate vectors because we don't introduce variables here.
        It plays role as the environment.
        current state consists of current word context
        Args:
        overall_state: 1, 1,  hidden_dim
        hidden_state: 1, 1, hidden_dim
        current_input: 1, word_dim*2
        '''
        self.observation['current_context'] = self.contexts[0][pos].view(1, -1)#1, hidden_size
        self.observation['current_hidden'] = self.hidden_state[0]#1, hidden_size
        self.observation['current_cell'] = self.cell_state[0]#1, hidden_size
        self.observation['current_word'] = current_word#1, word_dim
        self.observation['target'] = target#1, word_dim
        self.observation['current_pos'] = current_pos#1, word_dim
        self.observation['target_pos'] = target_pos#1, word_dim
        #Note, for policy network, state is generated by the environment, no gadients propagation
        with torch.no_grad():
            return self.observation
    
    def update_state(self, current_word, current_action):
        '''
        Update  current output and hidden state after the action
        Args:
        current_word: 1*emb_dim
        current_action: 0 or 1
        pre_cell_state: 1*hidden_dim
        pre_hidden_state: 1, 
        '''
        if current_action == 0:#Delete the word, remain as previous state
            current_hidden_state, current_cell_state = self.hidden_state, self.cell_state
        #######Note, biLSTM cannot use here because it needs  to read the whole sentence for backward layer
        else:#Use RNN to compute new output and new state
            h_c = (self.hidden_state, self.cell_state)
            _, (current_hidden_state, current_cell_state) = self.rnn(current_word.view(1, 1, -1), h_c)
        self.hidden_state, self.cell_state = current_hidden_state, current_cell_state
        return current_hidden_state, current_cell_state
    
    
class Policy_network(nn.Module):
    def __init__(self, config):
        '''
        Policy network, predict an action according to the current state
        A concept of pointer network
        '''
        super(Policy_network, self).__init__()
        #Map state into action probability
        self.config = config
        
        self.actions = config.actions
        self.action_num = len(self.actions)
        self.action_ids = list(range(len(self.actions)))
        #Functions, with parameters to be learned
        self.activation = nn.Tanh()
        self.enc2temp = nn.Linear(config.hidden_dim, 128)
        self.hidden2temp = nn.Linear(config.hidden_dim, 128)
        self.target2temp = nn.Linear(config.embed_dim, 128)
        self.state2action = nn.Linear(128, self.action_num)
       
        
    def forward(self, observation):
        '''
        observation, a dictionary
        '''
        h_e = observation['current_context']
        h_c = observation['current_hidden']
        target = observation['target']

        temp = self.enc2temp(h_e) + self.hidden2temp(h_c) + self.target2temp(target)
        temp = self.activation(temp)
        output = self.state2action(temp)
        action_prob = F.softmax(output, 1)
        return action_prob
        
class Agent(nn.Module):
    def __init__(self, config):
        '''
        Agent in a reinforcement learning
        The state: rnn output of a sentence, current hidden state, current word, target
        '''
        super(Agent, self).__init__()
        self.config = config
        self.actions = config.actions
        self.action_num = len(self.actions)
        self.action_ids = list(range(len(self.actions)))
        
        self.env = Env(config)
        self.pnet = Policy_network(config)
        
        if config.if_gpu: 
            self.env = self.env.cuda()
            self.pnet = self.pnet.cuda()
        
        self.vec2label = nn.Linear(config.hidden_dim, config.label_num)

        #Positional encoding
        n_position = 100
        self.position_enc = nn.Embedding(n_position, config.embed_dim, padding_idx=0)
        self.position_enc.weight.data = self.position_encoding_init(n_position, config.embed_dim)

        #Variable, hyperparameter to balance 
        self.gamma = 0.15#torch.tensor(0.5, requires_grad=True)
    


    def get_pos_emb(self, sents):
        '''
        Get position information for a sequence
        '''
        word_seq_embs = []
        for sent in sents:
            sent_len = len(sent)
            word_seq = torch.LongTensor(list(range(sent_len)))
            if self.config.if_gpu: word_seq = word_seq.cuda()
            word_seq_emb = self.position_enc(word_seq)
            word_seq_embs.append(word_seq_emb)
        return torch.stack(word_seq_embs)
    

    def forward(self, sents, masks, labels, texts):
        '''
        One sentence each time, note mask embedding is considered
        Args:
        sents: 1, max_len, emb_dim+mask_dim
        masks: 1, max_len
        labels:[1]
        '''

        #########Get the final output of the lstm
        #encode the whole sentence
        self.env.init_env(sents, masks, texts)

        ##########Get target embedding, average
        targets = get_target_emb(sents, masks)
        if self.config.if_gpu: targets = targets.cuda()

        ############Get position embedding: 1, max_len, word_emb
        sent_pos_emb = self.get_pos_emb(sents)
        #Get average position encoding: 1, word_emb
        target_pos_emb = get_target_emb(sent_pos_emb, masks)

        ############Only one sentence will be processed each time
        sent = sents[0]
        sent_pos_emb = sent_pos_emb[0]
        actions = []
        action_loss = []
        rewards = []
        actions_critic = []
        rewards_critic = []
        #Select each word
        #Only one sentence, so the length is real
        for i, word in enumerate(sent):
            #word embedding: 1, emb_dim
            word = word.view(1, -1)
            #word position embedding: 1, emb_dim
            pos_emb = sent_pos_emb[i].view(1, -1)
            
            #####*********No positional embedding*******
            word = word + pos_emb
            ###target embedding:1, emb_dim
            ###target pos embedding:1, emb_dim
            targets = targets + target_pos_emb

            
            #Get the state, not differentiable for the action
            observation = self.env.set_current_state(i, word, pos_emb, targets, target_pos_emb)
            
            
            p = self.pnet(observation)#p is a tensor, 1*num

            
            m = Categorical(p)
            action = m.sample()#sample an action
            
            reward = self.env.generate_reward(action.item(), i)
            step_loss = -m.log_prob(action)
            rewards.append(reward)
            
            #Record critic action and rewards
            action_critic, reward_critic = self.env.generate_critic_action(i)
            actions_critic.append(action_critic)
            rewards_critic.append(reward_critic)
   
            #Record action id and its corresponding probability
            action_loss.append(step_loss)
            actions.append(action.item())
            
            #Update internal state, hidden_state(1, 1, hidden_size)
            self.env.update_state(word, action.item())
        ###########output log probability, 1*hidden_dim
        hidden_state = self.env.hidden_state
        final_h = F.dropout(hidden_state[0], p=0.2, training=self.training)
        pred = F.log_softmax(self.vec2label(final_h), 1)#1*num


        #########loss of predict actions
        action_loss, classification_loss = self.calculate_action_loss(pred, labels, action_loss, rewards, actions_critic, rewards_critic)
        return pred, actions, action_loss, classification_loss


    def predict(self, sents, masks):
        '''
        one sentence each  time
        Args:
        sents: 1, max_len, emb_dim
        targets: 1, emb_dim
        '''
        #########Get the final output of the lstm
        #encode the whole sentence
        self.env.init_env(sents, masks)

        ##########Get target embedding, average
        targets = get_target_emb(sents, masks)

        ############Get position embedding: 1, max_len, word_emb
        sent_pos_emb = self.get_pos_emb(sents)
        #Get average position encoding: 1, word_emb
        target_pos_emb = get_target_emb(sent_pos_emb, masks)

        ############Only one sentence will be processed each time
        sent = sents[0]
        sent_pos_emb = sent_pos_emb[0]
        actions = []
        action_loss = []
        
        #Select each word
        for i, word in enumerate(sent):
            #word embedding: 1, emb_dim
            word = word.view(1, -1)
            #word position embedding: 1, emb_dim
            pos_emb = sent_pos_emb[i].view(1, -1)
            word = word + pos_emb
            #target embedding:1, emb_dim
            #target pos embedding:1, emb_dim
            targets = targets + target_pos_emb

            #Get the state, not differentiable for the action
            observation = self.env.set_current_state(i, word, pos_emb, targets, target_pos_emb)
            
            p = self.pnet(observation)#p is a tensor, 1*num
            
            action = p[0].argmax()#choose the action has largest probability

            actions.append(action.item())
            
            #Update internal state, hidden_state(1, 1, hidden_size)
            self.env.update_state(word, action.item())
        ###########output log probability, 1*hidden_dim
        hidden_state = self.env.hidden_state
        final_h = hidden_state[0]
        pred = F.log_softmax(self.vec2label(final_h), 1)#1*num

        pred_label = pred.argmax(1)
        return pred_label, actions

    def log_loss(self, preds, labels):
        '''
        Calculate the log_prob for the gold label
        logP(y=cg|X)
        '''
        loss = nn.NLLLoss()
        neg_log_prob = loss(preds, labels)#the smaller the better, positive
        return neg_log_prob


    def calculate_action_loss(self, preds, labels, action_loss, rewards, actions_critic, rewards_critic):
        '''
        Calculate reward values for a sentence
        preds: probability, [1, label_size]
        labels: a list of labels, [1]
        actions:[1, sent_len]
        select_action_probs:[1, sent_len]
        parse_weights:[1, sent_len], weights for each word given the target
        '''

        #loss for the gold ground, a python scalar
        classification_loss = self.log_loss(preds, labels)
        ground_truth_prob = -classification_loss.detach().cpu().numpy()#negative, the bigger the better
        
        
        #Loss for the prediction of actions, no gradients
        #Normalize the rewards
        rewards = torch.FloatTensor(rewards)#no gradient
        rewards_critic = torch.FloatTensor(rewards_critic)
        
        #gradients are passed in action_loss
        action_loss = torch.stack(action_loss, 1)
        #rewards = (rewards - rewards.mean())/rewards.var()
        #print('Rewards', rewards)
        #rewards python scalar
        if self.config.if_gpu:
            rewards = rewards.cuda()
            action_loss = action_loss.cuda()
        action_loss_avg = (action_loss*rewards).mean()
        #action_loss_avg = action_loss.mean()
        total_reward = ground_truth_prob
  
        action_loss = action_loss_avg * (2+total_reward)#smaller is better

        return action_loss, classification_loss


    ##Note this is a policy network, namely to use a network to predict actions
    def generate_action(self, p, training=True):
        '''
        Sample an action according to the probability
        Args:
        p: tensor, [action_num]
        '''
        p = p.detach().cpu().numpy()

        assert self.action_num == len(p)
        if training:
            #Consider exploration and exploitation
            action_id = np.random.choice(self.action_ids, p=p)
#             t = np.random.rand()
#             if t > self.config.epsilon:
#                 action_id = np.random.choice(self.action_ids, p=p)
#             else:#Exploration
#                 action_id = np.random.choice(self.action_ids)
        else:#In testing part, just choose the max prob one
            action_id = np.argmax(p)
        return action_id

    def assign_random_action(self):
        '''
        Assign a random action to the agent
        '''
        return np.random.choice(self.action_ids)


    

    def position_encoding_init(self, n_position, emb_dim):
        ''' Init the sinusoid position encoding table '''

        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
        

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
        pos_emb = torch.from_numpy(position_enc).type(torch.FloatTensor)
        if self.config.if_gpu: pos_emb = pos_emb.cuda()
        return pos_emb








    