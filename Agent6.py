import numpy as np
import torch
#from config import config
import torch.nn as nn
import torch.nn.functional as F
from parse_path import constituency_path, dependency_path
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

class Env(nn.Module):
    def __init__(self, config):
        '''
        Construct an virtual environment that can generate, update state and rewards to the agent
        '''
        super(Env, self).__init__()
        self.config = config
        
        
        self.observation = {}
        
        #output states of RNNs
        self.cell_state = None
        self.hidden_state = None
        self.contexts = None
    
        #Encode the entire sentence
        self.birnn = nn.LSTM(config.embed_dim, int(config.hidden_dim/2), batch_first=True, num_layers= config.layer_num, bidirectional=True, dropout=config.rnn_dropout)
        #Compress the sentence
        self.rnn = nn.LSTM(config.embed_dim, config.hidden_dim, batch_first=True, num_layers= config.layer_num, bidirectional=False, dropout=config.rnn_dropout)
        
        
    def forward(self):
        pass
    
    
    def step(self, action):
        observation = None
        reward = None
        done = None
        return observation, reward, done
        
    
    
    def generate_reward(self, preds, labels, actions, action_probs, parse_weights):
        #Final reward
        delete_num = len(actions) - sum(actions)
        #loss for the gold ground
        classification_loss = self.log_loss(preds, labels)
        ground_truth_prob = -classification_loss.detach().numpy()#negative, the bigger the better
        #print('Log_prob:', log_prob)
        #Reward the action that can remove words
        del_reward = float(delete_num)/2#len(actions)#more words deleted, better
        parse_reward = np.sum(np.array(actions) * parse_weights)
        reward = ground_truth_prob*3 + del_reward + parse_reward
        return reward


    
    def log_loss(self, preds, labels):
        '''
        Calculate the log_prob for the gold label
        logP(y=cg|X)
        '''
        loss = nn.NLLLoss()
        neg_log_prob = loss(preds, labels)#the smaller the better, positive
        return neg_log_prob
    
    def init_hidden(self):
        '''
        Initialize hidden state, C, H
        '''
        size = int(self.config.hidden_dim)
        h = torch.zeros(1, 1, size)
        c = torch.zeros(1, 1, size)
        self.hidden_state, self.cell_state = h, c

    
    def init_rnn_state(self, texts):
        '''
        Get the hidden state for the original sentence
        Args:
        texts: a tensor of texts, batch_size, max_len, embed_dim
        '''
        # #1, max_len, hidden_dim
        _, (h, c) = self.birnn(texts)
        #1, 1, hidden_dim
        batch_size = texts.size(0)
        h = h.view(batch_size, 1, -1)
        c = c.view(batch_size, 1, -1)
        #overall_state = h.view(batch_size, -1)
        self.hidden_state, self.cell_state = h, c
        
    def get_sent_context_state(self, texts):
        '''
        Get the hidden state for the original sentence
        Args:
        texts: a tensor of texts, batch_size, max_len, embed_dim
        '''
        # #1, max_len, hidden_dim
        contexts, _ = self.birnn(texts)
        #1, 1, hidden_dim
        self.contexts = contexts
        
        
    
    #Note the state should be non-differentiable
    def get_current_state(self, pos, current_word, current_pos, target, target_pos):
        '''
        Compute current state, just concatenate vectors because we don't introduce variables here.
        It plays role as the environment.
        current state consists of current word context
        Args:
        overall_state: 1, 1,  hidden_dim
        hidden_state: 1, 1, hidden_dim
        current_input: 1, word_dim*2
        '''
        self.observation['current_context'] = self.contexts[0][pos].view(1, -1).detach()#1, hidden_size
        self.observation['current_hidden'] = self.hidden_state[0].detach()#1, hidden_size
        self.observation['current_cell'] = self.cell_state[0].detach()#1, hidden_size
        self.observation['current_word'] = current_word.detach()#1, word_dim
        self.observation['target'] = target.detach()#1, word_dim
        self.observation['current_pos'] = current_pos.detach()#1, word_dim
        self.observation['target_pos'] = target_pos.detach()#1, word_dim
        #Note, for policy network, state is generated by the environment, no gadients propagation
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
        #Functions
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
            word_seq_emb = self.position_enc(word_seq)
            word_seq_embs.append(word_seq_emb)
        return torch.stack(word_seq_embs)
    

    def forward(self, sents, masks, labels, texts):
        '''
        One sentence each time
        Args:
        sents: 1, max_len, emb_dim
        masks: 1, max_len
        labels:[1]
        '''

        #########Get the final output of the lstm
        #encode the whole sentence
        self.env.init_hidden()
        self.env.get_sent_context_state(sents)

        ##########Get target embedding, average
        targets = get_target_emb(sents, masks)
        max_len = sents.size(1)
        weights = self.get_parsing_weights(texts, masks, max_len)
        #weights = weights.detach()

        ############Get position embedding: 1, max_len, word_emb
        sent_pos_emb = self.get_pos_emb(sents)
        #Get average position encoding: 1, word_emb
        target_pos_emb = get_target_emb(sent_pos_emb, masks)

        ############Only one sentence will be processed each time
        sent = sents[0]
        sent_pos_emb = sent_pos_emb[0]
        actions = []
        action_probs = []
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
            observation = self.env.get_current_state(i, word, pos_emb, targets, target_pos_emb)
            p = self.pnet(observation)#p is a tensor, 1*num
            action_id = self.generate_action(p[0], True)
            action_prob = p[0][action_id]
            #Record action id and its corresponding probability
            action_probs.append(action_prob)
            actions.append(action_id)
            
            #Update internal state, hidden_state(1, 1, hidden_size)
            self.env.update_state(word, action_id)
        ###########output log probability, 1*hidden_dim
        hidden_state = self.env.hidden_state
        final_h = F.dropout(hidden_state[0], p=0.5, training=self.training)
        pred = F.log_softmax(self.vec2label(final_h), 1)#1*num


        #########loss of predict actions
        action_loss, classification_loss = self.calculate_action_loss(pred, labels, actions, action_probs, weights)
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
        self.env.init_hidden()
        self.env.get_sent_context_state(sents)

        ##########Get target embedding, average
        targets = get_target_emb(sents, masks)
        max_len = sents.size(1)

        ############Get position embedding: 1, max_len, word_emb
        sent_pos_emb = self.get_pos_emb(sents)
        #Get average position encoding: 1, word_emb
        target_pos_emb = get_target_emb(sent_pos_emb, masks)

        ############Only one sentence will be processed each time
        sent = sents[0]
        sent_pos_emb = sent_pos_emb[0]
        actions = []
        action_probs = []
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
            observation = self.env.get_current_state(i, word, pos_emb, targets, target_pos_emb)
            p = self.pnet(observation)#p is a tensor, 1*num
            action_id = self.generate_action(p[0], False)
            
            actions.append(action_id)
            
            #Update internal state, hidden_state(1, 1, hidden_size)
            self.env.update_state(word, action_id)
        ###########output log probability, 1*hidden_dim
        hidden_state = self.env.hidden_state
        final_h = F.dropout(hidden_state[0], p=0.5, training=self.training)
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


    def calculate_action_loss(self, preds, labels, actions, action_probs, parse_weights):
        '''
        Calculate reward values for a sentence
        preds: probability, [1, label_size]
        labels: a list of labels, [1]
        actions:[1, sent_len]
        select_action_probs:[1, sent_len]
        parse_weights:[1, sent_len], weights for each word given the target
        '''
        #Final reward
        delete_num = len(actions) - sum(actions)
        #loss for the gold ground
        classification_loss = self.log_loss(preds, labels)
        ground_truth_prob = -classification_loss.detach().numpy()#negative, the bigger the better

        #Reward the action that can remove words
        del_reward = float(delete_num)/2#len(actions)#more words deleted, better
        parse_reward = np.sum(np.array(actions) * parse_weights)
            
        reward = ground_truth_prob + del_reward + parse_reward
        
        
        #Loss for the prediction of actions
        loss = -torch.log(torch.stack(action_probs)).mean()
  
        action_loss = loss * reward#smaller is better

        return action_loss, classification_loss


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

    def get_parsing_weights(self, sents, masks, max_len):
        '''
        Get sentence's parsing weights according to the target
        Args:
        sents: [batch_size, len], texts
        masks: [batch_size, len], binary
        '''
        ##########Get parsing weights for each sentence given a target
        target_indice = convert_mask_index(masks)#Get target indice
        weights = get_context_weight(sents, target_indice, max_len)
        #Not differentiable
        return weights
    

    def position_encoding_init(self, n_position, emb_dim):
        ''' Init the sinusoid position encoding table '''

        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
        

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
        return torch.from_numpy(position_enc).type(torch.FloatTensor)








    