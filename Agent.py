import numpy as np
import torch
#from config import config
import torch.nn as nn
import torch.nn.functional as F
class Agent(nn.Module):
    def __init__(self, config):
        '''
        Agent in a reinforcement learning
        '''
        super(Agent, self).__init__()
        self.config = config
        self.actions = config.actions
        self.action_num = len(self.actions)
        self.action_ids = list(range(len(self.actions)))
        #Map state into action probability
        self.state2temp = nn.Linear(config.hidden_dim*2+2*config.embed_dim, 256)
        self.temp2action = nn.Linear(256, self.action_num)
        self.activation = nn.Sigmoid()
        self.state2action = nn.Sequential(self.state2temp, self.activation, self.temp2action, self.activation)

        #Lstm
        self.rnn = nn.LSTM(config.embed_dim, int(config.hidden_dim), batch_first=True, num_layers= config.layer_num, bidirectional=False, dropout=config.rnn_dropout)
        self.rnn_critic = nn.LSTM(config.embed_dim, int(config.hidden_dim), batch_first=True, num_layers= config.layer_num, bidirectional=False, dropout=config.rnn_dropout)

        self.state2label = nn.Linear(config.hidden_dim, config.label_num)

        #Variable, hyperparameter to balance 
        self.gamma = 0.15#torch.tensor(0.5, requires_grad=True)
    
    def init_hidden(self):
        '''
        Initialize hidden state, C, H
        '''
        size = int(self.config.hidden_dim)
        h = torch.zeros(1, 1, size)
        c = torch.zeros(1, 1, size)
        return (h, c)

    def forward(self, sents, targets, labels):
        '''
        Args:
        sents: 1, max_len, emb_dim
        targets: 1, emb_dim
        labels:[1]
        '''
        h, c = self.init_hidden()
        cell_state = c #1,1,hidden_size
        hidden_state = h #1,1,hidden_size
        #Only one sentence will be processed each time
        sent = sents[0]
        actions = []
        select_action_probs = []
        for word in sent:
            word = word.view(1, -1)

            #state: 1*size
            word_target = torch.cat([word, targets], 1)
            state = self.compute_current_state(cell_state, hidden_state, word_target)

            p = self.predict_action_prob(state)#p is a tensor, 1*num
            action_id = self.generate_action(p[0], True)
            action_prob = p[0][action_id]
            #Record action id and its corresponding probability
            select_action_probs.append(action_prob)
            actions.append(action_id)
            #Update internal state, hidden_state(1, 1, hidden_size)
            hidden_state, cell_state = self.update_state(word, action_id, hidden_state, cell_state)
        #output log probability
        final_h = F.dropout(hidden_state[0], p=0.5, training=self.training)
        pred = F.log_softmax(self.state2label(final_h), 1)
        #final reward
        loss = self.calculate_loss(pred, labels, actions, select_action_probs)
        return pred, actions, loss

    def critic(self, sents, targets, labels):
        '''
        This function create a baseline for the agent
        Args:
        sents: 1, max_len, emb_dim
        targets: 1, emb_dim
        labels:[1]
        '''
        h, c = self.init_hidden()
        cell_state = c #1,1,hidden_size
        hidden_state = h #1,1,hidden_size
        #Only one sentence will be processed each time
        sent = sents[0]
        actions = []
        select_action_probs = []
        for word in sent:
            word = word.view(1, -1)

            # #state: 1*size
            # word_target = (word+targets)/2
            # state = self.compute_current_state(cell_state, hidden_state, word_target)
            # p = self.predict_action_prob(state)#p is a tensor, 1*num
            # #This is baseling
            # action_id = self.generate_action(p[0], False)
            # action_prob = p[0][action_id]
            # #Record action id and its corresponding probability
            # select_action_probs.append(action_prob)
            # actions.append(action_id)
            # #Update internal state, hidden_state(1, 1, hidden_size)
            _, (hidden_state, cell_state) = self.rnn_critic(word, (hidden_state, cell_state))
        #output log probability
        final_h = hidden_state[0]
        pred = F.log_softmax(self.state2label(final_h), 1)
        #final reward
        loss = self.calculate_loss(pred, labels, actions, select_action_probs)
        return pred, loss

    def predict(self, sents, targets):
        '''
        Args:
        sents: 1, max_len, emb_dim
        targets: 1, emb_dim
        '''
        hidden_state, cell_state = self.init_hidden()
        #Only one sentence will be processed each time
        sent = sents[0]
        actions = []
        select_action_probs = []
        for word in sent:
            word = word.view(1, -1)

            #state: 1*size
            word_target = torch.cat([word, targets], 1)
            state = self.compute_current_state(hidden_state, cell_state, word_target)
            p = self.predict_action_prob(state)#p is a tensor, 1*num
            action_id = self.generate_action(p[0], False)
            action_prob = p[0][action_id]
            #Record action id and its corresponding probability
            select_action_probs.append(action_prob)
            actions.append(action_id)
            #Update internal state, hidden_state(1, 1, hidden_size)
            hidden_state, cell_state = self.update_state(word, action_id, hidden_state, cell_state)
        #output log probability, 1*label_num
        pred = F.log_softmax(self.state2label(hidden_state[0]), 1)
        pred_label = pred.argmax(1)
        return pred_label, actions

    def get_ground_log_loss(self, preds, labels):
        '''
        Calculate the log_prob for the gold label
        logP(y=cg|X)
        '''
        loss = nn.NLLLoss()
        neg_log_prob = loss(preds, labels)#the smaller the better, positive
        return neg_log_prob


    def calculate_loss(self, preds, labels, actions, select_action_probs):
        '''
        Calculate reward values for a sentence
        preds: probability, [1, label_size]
        labels: a list of labels, [1]
        actions:[1, sent_len]
        select_action_probs:[1, sent_len]
        '''
        #Final reward
        delete_num = len(actions) - sum(actions)
        #loss for the gold ground
        loss = self.get_ground_log_loss(preds, labels)#smaller, the better
        #print('Log_prob:', log_prob)
        #Reward the action that can remove words
        reward = float(delete_num)/len(actions)#more words deleted, better
  
        output_loss = 2 * loss * (0.001+reward)#smaller is better
        #print('Reward:', reward)
        #loss in the process
        #log_select_action_probs = [item.log() for item in select_action_probs]
        #reward plus average log probability
        #reward += sum(log_select_action_probs)/len(actions)
        return output_loss


    def predict_action_prob(self, state):
        '''
        Predict probabilities of actions
        Args:
        State: [1, hidden_dim*2+emb_dim*2]
        Output:
        [1, action_num]
        '''
        action_probs = F.softmax(self.state2action(state), 1)
        return action_probs

    ##Note this is a policy network, namely to use a network to predict actions
    def generate_action(self, p, training=True):
        '''
        Sample an action according to the probability
        Args:
        p: tensor, [action_num]
        '''
        p = p.detach().numpy()

        assert self.action_num == len(p)
        if training:
            #Consider exploration and exploitation
            t = np.random.rand()
            if t > self.config.epsilon:
                action_id = np.random.choice(self.action_ids, p=p)
            else:#Exploration
                action_id = np.random.choice(self.action_ids)
        else:#In testing part, just choose the max prob one
            action_id = np.argmax(p)
        return action_id

    def assign_random_action(self):
        '''
        Assign a random action to the agent
        '''
        return np.random.choice(self.action_ids)

    def update_state(self, current_word, current_action, pre_hidden_state, pre_cell_state):
        '''
        Update  current output and hidden state after the action
        Args:
        current_word: 1*emb_dim
        current_action: 0 or 1
        pre_cell_state: 1*hidden_dim
        pre_hidden_state: 1, 
        '''
        if current_action == 0:#Delete the word, remain as previous state
            current_hidden_state, current_cell_state = pre_hidden_state, pre_cell_state
        #######Note, biLSTM cannot use here because it needs  to read the whole sentence for backward layer
        else:#Use RNN to compute new output and new state
            h_c = (pre_hidden_state, pre_cell_state)
            _, (current_hidden_state, current_cell_state) = self.rnn(current_word.view(1, 1, -1), h_c)
        return current_hidden_state, current_cell_state

    def compute_current_state(self, pre_hidden_state, pre_cell_state, current_input):
        '''
        Compute current state
        Args:
        cell_state: 1, 1, hidden_dim
        hidden_state: 1, 1, hidden_dim
        current_input: 1, word_dim*2
        '''
        current_state = torch.cat([pre_hidden_state[0], pre_cell_state[0], current_input], 1)
        return current_state








    