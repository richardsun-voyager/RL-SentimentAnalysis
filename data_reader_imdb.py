import numpy as np
import pandas as pd
import torch
from stanfordcorenlp import StanfordCoreNLP
stanford_nlp = StanfordCoreNLP(r'../data/stanford-corenlp-full-2018-02-27')


import en_core_web_sm
nlp = en_core_web_sm.load()

class dataHelper:
    def __init__(self, config, is_training=True):
        '''
        This class is able to:
        1. Load datasets
        2. Split sentences into words
        3. Map words into Idx
        '''
        self.config = config

        # id map to instance
        self.id2label = ["positive", "negative"]
        self.label2id = {v:k for k,v in enumerate(self.id2label)}

        self.UNK = "unk"
        self.EOS = "<eos>"
        self.PAD = "<pad>"
        
        self.is_training = is_training
        self.max_word_num = 600

        # data
        self.data = None
        self.index = 0
        self.data_len = 0
        self.emb, _ = self.load_pretrained_word_emb(self.config.pretrained_embed_path)

    def read_csv_data(self,file_name):
        '''
        Read CSV data
        '''
        data = pd.read_csv(file_name, header=None, names=['label', 'text'])
        self.data = data
        self.data_len = len(data)
        #return data
    
    def stanford_tokenize(self, sent_str):
        return stanford_nlp.word_tokenize(sent_str)

    def clean_str(self, text):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def clean_text(self, text):
        # return word_tokenize(sent_str)
        if self.config.is_stanford_nlp:
            text = self.clean_str(text)
        sent_str = " ".join(text.split("-"))
        sent_str = " ".join(sent_str.split("/"))
        sent_str = " ".join(sent_str.split("("))
        sent_str = " ".join(sent_str.split(")"))
        sent_str = " ".join(sent_str.split())
        return sent_str

    
    def stanford_tokenize(self, sent_str):
        return stanford_nlp.word_tokenize(sent_str)


    def tokenize(self, sent_str):
        '''
        Split a sentence into tokens
        '''
        sent = nlp(sent_str)
        #return [item.text.lower() for item in sent]
        return [item.text for item in sent]
        #return tokenizer(sent_str)
        

    def get_local_word_embeddings(self, pretrained_word_emb, local_vocab):
        '''
        Obtain local word embeddings based on pretrained ones
        local_vocab: word in local vocabulary, in order
        '''
        local_emb = []
        #if the unknow vectors were not given, initialize one
        if self.UNK not in pretrained_word_emb.keys():
            pretrained_word_emb[self.UNK] = np.random.randn(self.config.embed_dim)
        for w in local_vocab:
            local_emb.append(self.word2vec(pretrained_word_emb, w))
        local_emb = np.vstack(local_emb)
        emb_path = self.config.embed_path
        if not os.path.exists(os.path.dirname(emb_path)):
            print('Path not exists')
            os.mkdir(os.path.dirname(emb_path))
        #Save the local embeddings
        with open(emb_path, 'wb') as f:
            pickle.dump(local_emb, f)
            print('Local Embeddings Saved!')
        return local_emb



    def load_pretrained_word_emb(self, file_path):
        '''
        Load a specified vocabulary
        '''
        word_emb = {}
        vocab_words = set()
        with open(file_path) as fi:
            for line in fi:
                items = line.split()
                word = ' '.join(items[:-1*self.config.embed_dim])
                vec = items[-1*self.config.embed_dim:]
                word_emb[word] = np.array(vec, dtype=np.float32)
                vocab_words.add(word)
        return word_emb, vocab_words

    def word2vec(self, vocab, word):
        '''
        Map a word into a vec
        '''
        try:
            vec = vocab[word]
        except:
            vec = vocab[self.UNK]
        return vec
    
    
    def generate_sample(self, data):
        '''
        Generate a batch of training dataset
        '''
        batch_size = self.config.batch_size
        select_index = np.random.choice(len(data), batch_size)
        select_data = data.iloc[select_index, :]
        return select_data
    
    def word_id_map(self, tokens, max_len):
        '''
        Generate a sequence of word embeddings
        '''
        dim = self.config.embed_dim
        vecs = [self.word2vec(self.emb, w) for w in tokens]
        #Padding with zero
        for i in np.arange(max_len - len(tokens)):
            vecs.append(np.zeros(dim))
        vecs = np.vstack(vecs)
        return vecs
    
    def decode_samples(self, samples):
        tokens = samples['text'].map(self.tokenize).values
        tokens = [sent if len(sent)<=self.max_word_num else sent[:self.max_word_num] for sent in tokens]
        sent_lens = torch.LongTensor(list(map(len, tokens)))
        max_len = max(sent_lens)
        sent_vecs = np.stack([self.word_id_map(sent, max_len) for sent in tokens])
        sent_vecs = torch.FloatTensor(sent_vecs)
        labels = torch.LongTensor(samples.label.values)
        #Sorting
        sent_lens, perm_idx = sent_lens.sort(0, descending=True)
        sent_vecs = sent_vecs[perm_idx]
        labels = labels[perm_idx]
        
        return sent_vecs, labels, sent_lens
    
    
    def get_ids_samples(self, is_balanced=False):
        if self.is_training:
            samples = self.generate_sample(self.data)
            #texts = samples['text']
            sent_vecs, labels, sent_lens = self.decode_samples(samples)
        else:
            if self.index == self.data_len:
                print('Testing Over!')
            #First get batches of testing data
            if self.data_len - self.index >= self.config.batch_size:
                #print('Testing Sample Index:', self.index)
                start = self.index
                end = start + self.config.batch_size
                samples = self.data.iloc[start: end, :]
                self.index += self.config.batch_size
                sent_vecs, labels, sent_lens = self.decode_samples(samples)

            else:#Then generate testing data one by one
                samples =  self.data_batch[self.index] 
                sent_vecs, labels, sent_lens = self.decode_samples(samples)
                self.index += 1
        return sent_vecs, labels, sent_lens
        
        
        
    

    