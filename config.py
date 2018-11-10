class Config:
    def __init__(self):
        '''
        Configuration
        '''
        self.actions = ['delete', 'select']
        self.action2id = {a:i for i, a in enumerate(self.actions)}
        self.id2action = {i:a for i, a in enumerate(self.actions)}
        #Word embeddings dimension
        self.embed_dim = 300#word embeddings
        self.embed_num = 5091 #Number of words
        self.mask_dim = 50
        #LSTM hidden state dimension
        
        
        
        #rnn    
        self.hidden_dim = 256
        self.layer_num = 1
        self.rnn_dropout = 0.0
        self.clip_norm = 3

        #Exploration ratio
        self.epsilon = 0.05
        self.label_num = 3

        #path
        self.data_path = 'data/restaurant/'
        self.dic_path = "data/restaurant/vocab/dict.pkl"
        self.embed_path = "data/restaurant/vocab/local_emb.pkl"
        self.model_path = "data/models/"
        self.log_path = 'data/logs/'
        self.pretrained_embed_path = "../data/glove.840B.300d.txt"

        self.train_path = "data/restaurant/train.pkl"
        self.valid_path = "data/restaurant/valid.pkl"
        self.test_path = "data/restaurant/test.pkl"

        #training optimizer
        self.batch_size = 1
        self.opt = 'Adam'
        self.dropout = 0.5
        self.if_update_embed = False
        self.if_gpu = False
        self.lr = 0.0001
        self.l2 = 0.0001
        self.epochs= 100
        self.adjust_every = 8

        #tokenization tool
        self.is_stanford_nlp = True


    def __repr__(self):
        return str(vars(self))

config = Config()