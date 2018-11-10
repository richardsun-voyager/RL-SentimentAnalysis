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
        super(Agent, self).__init__()
        self.config = config
        
        
    def create_state(self):
        pass
    
    def update_state(self, action, current_state):
        pass
    
    def generate_reward(self, action):
        pass
    
    

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
        
        #functions to generate state
        self.hidden_state2temp = nn.Linear(config.hidden_dim*2, 128)
        self.word_emb2temp = nn.Linear(config.embed_dim*2, 128)
        
        
        #Map state into action probability
        self.state2temp = nn.Linear(config.hidden_dim*2+2*config.embed_dim, 256)
        self.temp2out = nn.Linear(256, self.action_num)
        self.activation = nn.Sigmoid()
        self.state2action = nn.Sequential(self.state2temp, nn.Tanh(), self.temp2out, nn.Sigmoid())

        #Lstm
        self.rnn = nn.LSTM(config.embed_dim, int(config.hidden_dim), batch_first=True, num_layers= config.layer_num, bidirectional=False, dropout=config.rnn_dropout)

        self.birnn = nn.LSTM(config.embed_dim, int(config.hidden_dim/2), batch_first=True, num_layers= config.layer_num, bidirectional=True, dropout=config.rnn_dropout)
        
        #

        self.state2label = nn.Linear(config.hidden_dim, config.label_num)

        #Positional encoding
        n_position = 100
        self.position_enc = nn.Embedding(n_position, config.embed_dim, padding_idx=0)
        self.position_enc.weight.data = self.position_encoding_init(n_position, config.embed_dim)

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
    
    def get_overall_final_state(self, texts):
        '''
        Get the hidden state for the original sentence
        Args:
        texts: a tensor of texts, batch_size, max_len, embed_dim
        '''
        # #1, max_len, hidden_dim
        _, (h, c) = self.rnn(texts)
        #1, 1, hidden_dim
        batch_size = texts.size(0)
        h = h.view(batch_size, 1, -1)
        c = c.view(batch_size, 1, -1)
        #overall_state = h.view(batch_size, -1)
        return h, c


    def forward(self, sents, masks, labels, texts):
        '''
        One sentence each time
        Args:
        sents: 1, max_len, emb_dim
        masks: 1, max_len
        labels:[1]
        '''

        #########Initialize hidden state
        h, c = self.init_hidden()
        cell_state = c #1,1,hidden_size
        hidden_state = h #1,1,hidden_size

        #########Get the final output of the lstm
        #encode the whole sentence
        hidden_state, cell_state = self.get_overall_final_state(sents)


        ##########Get target embedding, average
        targets = get_target_emb(sents, masks)

        ##########Get parsing weights for each sentence given a target
        target_indice = convert_mask_index(masks)#Get target indice
        max_len = sents.size(1)
        weights = get_context_weight(texts, target_indice, max_len)
        #Not differentiable
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

            #state: 1*size
            word_target = torch.cat([word, targets], 1)
            state = self.compute_current_state(cell_state, hidden_state, word_target)

            p = self.predict_action_prob(state)#p is a tensor, 1*num
            action_id = self.generate_action(p[0], True)
            #action_prob = p[0][action_id]
            #Record action id and its corresponding probability
            action_probs.append(p[0])
            actions.append(action_id)
            #Update internal state, hidden_state(1, 1, hidden_size)
            hidden_state, cell_state = self.update_state(word, action_id, hidden_state, cell_state)
        ###########output log probability, 1*hidden_dim
        final_h = F.dropout(hidden_state[0], p=0.5, training=self.training)
        pred = F.log_softmax(self.state2label(final_h), 1)#1*num

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
        h, c = self.init_hidden()
        cell_state = c #1,1,hidden_size
        hidden_state = h #1,1,hidden_size

        #########Get the final output of the lstm
        #batch_size, hidden_dim
        hidden_state, cell_state = self.get_overall_final_state(sents)

        #Get target embedding, average
        targets = get_target_emb(sents, masks)

        #Get position embedding: 1, max_len, word_emb
        sent_pos_emb = self.get_pos_emb(sents)
        #Get average position encoding: 1, word_emb
        target_pos_emb = get_target_emb(sent_pos_emb, masks)


        #Only one sentence will be processed each time
        sent = sents[0]
        sent_pos_emb = sent_pos_emb[0]
        actions = []
        action_probs = []
        for i, word in enumerate(sent):
            #word embedding: 1, emb_dim
            word = word.view(1, -1)
            #word position embedding: 1, emb_dim
            pos_emb = sent_pos_emb[i].view(1, -1)
            word = word + pos_emb
            #target embedding:1, emb_dim
            #target pos embedding:1, emb_dim
            targets = targets + target_pos_emb

            #state: 1*size
            word_target = torch.cat([word, targets], 1)
            
            #State is generated by the environment, not differentiable
            state = self.compute_current_state(cell_state, hidden_state, word_target)

            p = self.predict_action_prob(state)#p is a tensor, 1*num
            action_id = self.generate_action(p[0], False)


            actions.append(action_id)
            #Update internal state, hidden_state(1, 1, hidden_size)
            hidden_state, cell_state = self.update_state(word, action_id, hidden_state, cell_state)
        #output log probability, 1*label_num
        pred = F.log_softmax(self.state2label(hidden_state[0]), 1)
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
        #print('Log_prob:', log_prob)
        #Reward the action that can remove words
        del_reward = float(delete_num)/2#len(actions)#more words deleted, better
        parse_reward = np.sum(np.array(actions) * parse_weights)
            
        reward = ground_truth_prob + del_reward + parse_reward
        
        #Loss for the prediction of actions
        action_probs = torch.log(torch.stack(action_probs))
        actions = torch.LongTensor(actions)
        loss = self.log_loss(action_probs, actions)
  
        action_loss = loss * reward#smaller is better
        #print('Reward:', reward)
        #loss in the process
        #log_select_action_probs = [item.log() for item in select_action_probs]
        #reward plus average log probability
        #reward += sum(log_select_action_probs)/len(actions)
        return action_loss, classification_loss


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

    #Note the state should be non-differentiable
    def compute_current_state(self, pre_cell_state, pre_hidden_state, current_input):
        '''
        Compute current state, just concatenate vectors because we don't introduce variables here.
        It plays role as the environment.
        Args:
        overall_state: 1, 1,  hidden_dim
        hidden_state: 1, 1, hidden_dim
        current_input: 1, word_dim*2
        '''
        intern_state = torch.cat([pre_cell_state[0], pre_hidden_state[0]], 1)
        #2*hidden_dim + 2 * word_dim
        
        current_state = torch.cat([intern_state, current_input], 1)
        #No gradient propagation, state is generated by the environment
        with torch.no_grad():
            return current_state
    

    def position_encoding_init(self, n_position, emb_dim):
        ''' Init the sinusoid position encoding table '''

        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
        

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
        return torch.from_numpy(position_enc).type(torch.FloatTensor)








    