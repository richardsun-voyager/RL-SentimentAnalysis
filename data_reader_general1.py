#############################################
#This file aims to preprocess data and generate samples for training and testing
from collections import namedtuple, defaultdict
import codecs
from config import config
from bs4 import BeautifulSoup
import pdb
import torch
#import tokenizer
import numpy as np
import re
import pickle
import random
import os
from collections import Counter
##Added by Richard Sun
from allennlp.modules.elmo import Elmo, batch_to_ids
import en_core_web_sm
nlp = en_core_web_sm.load()

options_file = "../data/Elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "../data/Elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo = Elmo(options_file, weight_file, 2, dropout=0)

SentInst = namedtuple("SentenceInstance", "id text text_ids text_inds opinions")
OpinionInst = namedtuple("OpinionInstance", "target_text polarity class_ind target_mask target_ids")

class dataHelper():
    def __init__(self, config):
        '''
        This class is able to:
        1. Load datasets
        2. Split sentences into words
        3. Map words into Idx
        '''
        self.config = config

        # id map to instance
        self.id2label = ["positive", "neutral", "negative"]
        self.label2id = {v:k for k,v in enumerate(self.id2label)}

        self.UNK = "unk"
        self.EOS = "<eos>"
        self.PAD = "<pad>"

        # data
        self.train_data = None
        self.test_data = None

    def read_csv_data(self,file_name):
        '''
        Read CSV data
        '''
        import pandas as pd
        data = pd.read_csv(file_name)
        data_num = data.shape[0]
        sentence_list = []
        for i in np.arange(data_num):
            sent_id = i
            sent_text = data['Tweet'][i]
            opinion_list = []
            term = data['Keyword'][i]
            if pd.isnull(data['Sentiment'][i]):
                print('the '+str(i)+' empty')
                continue
            polarity = data['Sentiment'][i].lower()
            opinion_inst = OpinionInst(term, polarity, None, None, None)
            opinion_list.append(opinion_inst)
            sent_Inst = SentInst(sent_id, sent_text, None, None, opinion_list)
            sentence_list.append(sent_Inst)
        return sentence_list
        
    
    def read_xml_data(self, file_name):
        '''
        Read XML data
        '''
        f = codecs.open(file_name, "r", encoding="utf-8")
        soup = BeautifulSoup(f.read(), "lxml")
        sentence_tags = soup.find_all("sentence")
        sentence_list = []
        for sent_tag in sentence_tags:
            sent_id = sent_tag.attrs["id"]
            sent_text = sent_tag.find("text").contents[0]
            opinion_list = []
            try:
                asp_tag = sent_tag.find_all("aspectterms")[0]
            except:
                # print "{0} {1} has no opinions".format(sent_id, sent_text)
                #print(sent_tag)
                continue
            opinion_tags = asp_tag.find_all("aspectterm")
            for opinion_tag in opinion_tags:
                term = opinion_tag.attrs["term"]
                if term not in sent_text: pdb.set_trace()
                polarity = opinion_tag.attrs["polarity"]
                opinion_inst = OpinionInst(term, polarity, None, None, None)
                opinion_list.append(opinion_inst)
            sent_Inst = SentInst(sent_id, sent_text, None, None, opinion_list)
            sentence_list.append(sent_Inst)

        return sentence_list


    def tokenize(self, sent_str):
        '''
        Split a sentence into tokens
        '''
        # return word_tokenize(sent_str)
        sent_str = " ".join(sent_str.split("-"))
        sent_str = " ".join(sent_str.split("/"))
        sent_str = " ".join(sent_str.split("!"))
        #For indonesian
        sent_str = " ".join(sent_str.split("@"))
        sent = nlp(sent_str)
        return [item.text.lower() for item in sent]
        
    # namedtuple is protected!
    def process_raw_data(self, data, is_training=True):
        '''
        Tokenize each sentence, compute aspect mask for each sentence
        '''
        sent_len = len(data)
        #print('Sentences Num:', sent_len)
        text_words = []
        for sent_i in np.arange(sent_len):
            sent_inst = data[sent_i]
            #Tokenize texts
            sent_tokens = self.tokenize(sent_inst.text)
            text_words.append(sent_tokens)
            sent_inst = sent_inst._replace(text_inds = sent_tokens)
            opinion_list = []
            opi_len = len(sent_inst.opinions)
            #Read  opinion info
            for opi_i in np.arange(opi_len):
                opi_inst = sent_inst.opinions[opi_i]

                target = opi_inst.target_text
                target_tokens = self.tokenize(target)
                try:
                    target_start = sent_tokens.index(target_tokens[0])
                    target_end = sent_tokens[max(0, target_start - 1):].index(target_tokens[-1])  + max(0, target_start - 1)
                except:
                    #pdb.set_trace()
                    print('Error data:', sent_inst.text)
                    print('Target error ', target_tokens)
                    continue
                    
                if target_start < 0 or target_end < 0:
                    #pdb.set_trace()
                    print('Traget not in the vocabulary')
                    continue
                    
                mask = [0] * len(sent_tokens)
                for m_i in range(target_start, target_end + 1):
                    mask[m_i] = 1

                label = opi_inst.polarity
                if label == "conflict":  continue  # ignore conflict ones
                opi_inst = opi_inst._replace(class_ind = self.label2id[label])
                opi_inst = opi_inst._replace(target_mask = mask)
                opinion_list.append(opi_inst)
            
            sent_inst = sent_inst._replace(opinions = opinion_list)
            data[sent_i] = sent_inst
        #Map a token into an ID
        #data, words = self.text2ids(data, text_words, is_training)
        return data, text_words

    

    def text2ids(self, data, word2id, is_training):
        '''
        Map each word into an id in a text
        Args:
        data: namtuples
        texts: all the text
        '''
        #Build vocab


        def w2id(w):
            try:
                id = word2id[w]
            except:
                id = word2id[self.UNK]
            return id
        
        #Get the IDs for special tokens
        self.UNK_ID = w2id(self.UNK)
        self.PAD_ID = w2id(self.PAD)
        self.EOS_ID = w2id(self.EOS)
        sent_count = len(data)
        #print('Sentences Num:', sent_len)
        #Update nametuple
        for sent_i in np.arange(sent_count):
            sent_inst = data[sent_i]
            #Tokenize texts
            sent_tokens = sent_inst.text_inds
            #Map each token into an ID
            sent_ids = [w2id(token) for token in sent_tokens]
            sent_inst = sent_inst._replace(text_ids = sent_ids)
            #Read  opinion info
            opi_len = len(sent_inst.opinions)
            #Map target words into IDs
            for opi_i in np.arange(opi_len):
                opi_inst = sent_inst.opinions[opi_i]
                target = opi_inst.target_text
                target_tokens = self.tokenize(target)
                target_ids = [w2id(token) for token in target_tokens]
                opi_inst = opi_inst._replace(target_ids=target_ids)
                sent_inst.opinions[opi_i] = opi_inst
            data[sent_i] = sent_inst
        return data


    def build_local_vocab(self, texts, max_size):
        '''
        Build and save a vocabulary based on current texts
        texts: lists of words
        '''
        words = []
        for text in texts:
            words.extend(text)
        word_freq_pair = Counter(words)
        print('Tokenized Word Number:', len(word_freq_pair))
        if len(word_freq_pair) < max_size-3:
            max_size = len(word_freq_pair)
        
        most_freq = word_freq_pair.most_common(max_size-3)
        words, _ = zip(*most_freq)
        words = list(words)
        if self.UNK not in words:
            words.insert(0, self.UNK)
        if self.EOS not in words:
            words.append(self.EOS)
        if self.PAD not in words:
            words.append(self.PAD)
        #Build dictionary
        print('Local Vocabulary Size:', len(words))
        word2id = {w:i for i, w in enumerate(words)}
        id2word = {i:w for i, w in enumerate(words)}

        #Save the dictionary
        dict_file = config.dic_path
        dict_path = os.path.dirname(dict_file)  
        if not os.path.exists(dict_path):
            print('Dictionary path doesnot exist')
            print('Create...')
            os.mkdir(dict_path)
        with open(dict_file, 'wb') as f:
            pickle.dump([word2id, id2word, words], f)
            print('Dictionary created successfully')
        return words


    def get_local_word_embeddings(self, pretrained_word_emb, local_vocab):
        '''
        Obtain local word embeddings based on pretrained ones
        local_vocab: word in local vocabulary, in order
        '''
        local_emb = []
        #if the unknow vectors were not given, initialize one
        if self.UNK not in pretrained_word_emb.keys():
            pretrained_word_emb[self.UNK] = np.random.randn(config.embed_dim)
        for w in local_vocab:
            local_emb.append(self.word2vec(pretrained_word_emb, w))
        local_emb = np.vstack(local_emb)
        emb_path = config.embed_path
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
                word_emb[items[0]] = np.array(items[1:], dtype=np.float32)
                vocab_words.add(items[0])
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
    

    def read(self, data_path, data_source='xml', is_training=True):
        '''
        read and process raw data, create dictionary and index based on the training data
        '''
        if data_source == 'csv':
            train_data = self.read_csv_data(data_path)
        else:
            train_data = self.read_xml_data(data_path)
        #self.test_data = self.read_xml_data(test_data)
        print('Dataset number:', len(train_data))
        #print('Testing dataset number:', len(self.test_data))
        data = self.process_raw_data(train_data, is_training)

        # emb = self.load_pretrained_word_emb(config.pretrained_embed_path)
        # _ = self.get_local_word_embeddings(emb, words)
        #test_data = self.process_raw_data(self.test_data)
        return data
    
        

    # shuffle and to batch size
    def to_batches(self, data, if_batch = False):
        all_triples = []
        # list of list
        pair_couter = defaultdict(int)
        for sent_inst in data:
            tokens = sent_inst.text_inds
            token_ids = sent_inst.text_ids
            #print(tokens)
            for opi_inst in sent_inst.opinions:
                if opi_inst.polarity is None:  continue # conflict one
                mask = opi_inst.target_mask
                polarity = opi_inst.class_ind
                if tokens is None or mask is None or polarity is None: 
                    continue
                all_triples.append([tokens, mask, polarity, token_ids])
                pair_couter[polarity] += 1
                
        print(pair_couter)

        if if_batch:
            print('Shuffle')
            random.shuffle(all_triples)
            batch_n = int(len(all_triples) / self.config.batch_size + 1)
            print("{0} instances with {1} batches".format(len(all_triples), batch_n))
            ret_triples = []
            
            offset = 0
            for i in range(batch_n):
                start = self.config.batch_size * i
                end = min(self.config.batch_size * (i+1), len(all_triples) )
                ret_triples.append(all_triples[start : end])
            return ret_triples
        else:
            return all_triples




class data_reader:
    def __init__(self, config, is_training=True):
        '''
        Load dataset and create batches for training and testing
        '''
        self.is_training = is_training
        self.config = config
        self.dh = dataHelper(config)
        self.UNK = self.dh.UNK
        self.EOS = self.dh.EOS
        self.PAD = self.dh.PAD

    def read_train_test_data(self, data_path_list, data_source='xml', use_glove=False):
        '''
        Reading Raw Dataset from several files
        Args:
        data_path_list: a list of raw text files
        '''
        print('Reading Dataset....')
        data_list = []
        text_word_list = []
        for data_path in data_path_list:
            data, text_words = self.dh.read(data_path, data_source, self.is_training)
            data_list.append(data)
            text_word_list.extend(text_words)
        #Build dictionary based on all the words
        if not use_glove:
            #######################Buld a local vocabulary
            ## Create embeddings for the words in the given dataset
            words = self.dh.build_local_vocab(text_word_list, self.config.embed_num)
            #Create local embeddings for each word
            dict_file = config.dic_path
            #The dictionary must be created in advance
            if not os.path.exists(dict_file):
                print('Dictionary file not exist!')
            with open(dict_file, 'rb') as f:
                word2id, _, _ = pickle.load(f)
            emb, _ = self.dh.load_pretrained_word_emb(config.pretrained_embed_path)
            _ = self.dh.get_local_word_embeddings(emb, words)
        else:
            ####################Use Glove vocabulary
            emb, words = self.dh.load_pretrained_word_emb(config.pretrained_embed_path)
            #Save word embeddings in binary format
            _ = self.dh.get_local_word_embeddings(emb, words)
            #Build dictionary
            print('Glove Vocabulary Size:', len(words))
            word2id = {w:i for i, w in enumerate(words)}
        #Map words into Ids
        for i, data in enumerate(data_list):
            #Get file name
            name = data_path_list[i]
            name = os.path.basename(name)
            data = self.dh.text2ids(data, word2id, self.is_training)
            data_batch = self.dh.to_batches(data)
            #Save each processed data
            self.save_data(data_batch, self.config.data_path+name+'.pkl')

        # print('Preprocessing Dataset....')
        # self.data_batch = self.dh.to_batches(data)
        # self.data_len = len(self.data_batch)
        self.UNK_ID = self.dh.UNK_ID
        self.PAD_ID = self.dh.PAD_ID
        self.EOS_ID = self.dh.EOS_ID
        print('Preprocessing Over!')

    def read_raw_data(self, data_path, data_source='xml', use_glove=False):
        '''
        Reading Raw Dataset from one file
        '''
        print('Reading Dataset....')
        data, text_words = self.dh.read(data_path, data_source, self.is_training)
        #Build dictionary
        if not use_glove:
            words = self.dh.build_local_vocab(text_words, self.config.embed_num)
            #Create local embeddings
            emb, _ = self.dh.load_pretrained_word_emb(config.pretrained_embed_path)
            _ = self.dh.get_local_word_embeddings(emb, words)
            #Map words into Ids
            dict_file = config.dic_path
            #The dictionary must be created in advance
            if not os.path.exists(dict_file):
                print('Dictionary file not exist!')
            with open(dict_file, 'rb') as f:
                word2id, _, _ = pickle.load(f)
        else:
            ####################Use Glove vocabulary
            emb, words = self.dh.load_pretrained_word_emb(config.pretrained_embed_path)
            #Save word embeddings in binary format
            _ = self.dh.get_local_word_embeddings(emb, words)
            #Build dictionary
            print('Glove Vocabulary Size:', len(words))
            word2id = {w:i for i, w in enumerate(words)}
        data = self.dh.text2ids(data, word2id, self.is_training)
        print('Preprocessing Dataset....')
        self.data_batch = self.dh.to_batches(data)
        self.data_len = len(self.data_batch)
        self.UNK_ID = self.dh.UNK_ID
        self.PAD_ID = self.dh.PAD_ID
        self.EOS_ID = self.dh.EOS_ID
        print('Preprocessing Over!')

    def save_data(self, data, save_path):
        '''
        Save the data in specified folder
        '''
        try:
            with open(save_path, "wb") as f:
                pickle.dump(data,f)
            print('Saving successfully!')
        except:
            print('Saving failure!')   

    def load_data(self, load_path):
        '''
        Load the dataset
        '''
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                self.data_batch = pickle.load(f)
                self.data_len = len(self.data_batch)
            self.load_local_dict()
        else:
            print('Data not exist!')  
        return  self.data_batch

    def load_local_dict(self):
        '''
        Load dictionary files
        '''
        if not os.path.exists(config.dic_path):
            print('Dictionary file not exist!')
        with open(config.dic_path, 'rb') as f:
            word2id, _, _ = pickle.load(f)
        self.UNK_ID = word2id[self.dh.UNK]
        self.PAD_ID = word2id[self.dh.PAD]
        self.EOS_ID = word2id[self.dh.EOS]

    def split_save_data(self, train_path, valid_path):
        '''
        split dataset into training and validation parts
        '''
        np.random.shuffle(self.data_batch)
        train_num = int(self.data_len*5.0/6)
        training_batch = self.data_batch[:train_num]
        valid_batch = self.data_batch[train_num:]
        try:
            with open(train_path, "wb") as f:
                pickle.dump(training_batch,f)
            with open(valid_path, "wb") as f:
                pickle.dump(valid_batch,f)  
            print('Saving successfully!')
        except:
            print('Saving failure!')   
    


class data_generator:
    def __init__(self, config, data_batch, is_training=True):
        '''
        Generate training and testing samples
        Args:
        config: configuration parameters
        data_batch: data list, each contain a nametuple
        '''    
        self.is_training = is_training
        self.config = config
        self.index = 0
        self.data_batch = data_batch
        self.data_len = len(self.data_batch)
        self.UNK = "unk"
        self.EOS = "<eos>"
        self.PAD = "<pad>"
        self.load_local_dict()
        
    def remove_empty_target(self, data_batch):
        '''
        Remove items without targets
        '''
        data_batch = [item for item in data_batch if sum(item[1])>0]
        return data_batch

    def load_local_dict(self):
        '''
        Load dictionary files
        '''
        if not os.path.exists(config.dic_path):
            print('Dictionary file not exist!')
        with open(config.dic_path, 'rb') as f:
            word2id, _, _ = pickle.load(f)
        self.UNK_ID = word2id[self.UNK]
        self.PAD_ID = word2id[self.PAD]
        self.EOS_ID = word2id[self.EOS]

    def generate_sample(self, all_triples):
        '''
        Generate a batch of training dataset
        '''
        batch_size = self.config.batch_size
        select_index = np.random.choice(len(all_triples), batch_size)
        select_trip = [all_triples[i] for i in select_index]
        return select_trip


    def elmo_transform(self, triples):
        '''
        Transform sentences into elmo, each sentence represented by words
        '''
        token_list, mask_list, label_list, _ = zip(*triples)
        sent_lens = [len(tokens) for tokens in token_list]
        sent_lens = torch.LongTensor(sent_lens)
        label_list = torch.LongTensor(label_list)
        max_len = max(sent_lens)
        batch_size = len(sent_lens)
        character_ids = batch_to_ids(token_list)
        embeddings = elmo(character_ids)
        #batch_size*word_num * 1024
        sent_vecs = embeddings['elmo_representations'][0]
        #Padding the mask to same lengths
        mask_vecs = np.zeros([batch_size, max_len])
        mask_vecs = torch.LongTensor(mask_vecs)
        for i, mask in enumerate(mask_list):
            mask_vecs[i, :len(mask)] = torch.LongTensor(mask)
        return sent_vecs, mask_vecs, label_list, sent_lens

    

    def reset_samples(self):
        self.index = 0

    def pad_data(self, sents, masks, labels):
        '''
        Padding sentences to same size
        '''
        sent_lens = [len(tokens) for tokens in sents]
        sent_lens = torch.LongTensor(sent_lens)
        label_list = torch.LongTensor(labels)
        max_len = max(sent_lens)
        batch_size = len(sent_lens)
        #Padding mask
        mask_vecs = np.zeros([batch_size, max_len])
        mask_vecs = torch.LongTensor(mask_vecs)
        for i, mask in enumerate(masks):
            mask_vecs[i, :len(mask)] = torch.LongTensor(mask)
        #padding sent with PAD IDs
        sent_vecs = np.ones([batch_size, max_len]) * self.PAD_ID
        sent_vecs = torch.LongTensor(sent_vecs)
        for i, s in enumerate(sents):
            sent_vecs[i, :len(s)] = torch.LongTensor(s)
        sent_lens, perm_idx = sent_lens.sort(0, descending=True)
        sent_vecs = sent_vecs[perm_idx]
        mask_vecs = mask_vecs[perm_idx]
        label_list = label_list[perm_idx]
        return sent_vecs, mask_vecs, label_list, sent_lens

    def get_ids_samples(self, with_text=False):
        '''
        Get samples including ids of words, labels
        '''
        if self.is_training:
            samples = self.generate_sample(self.data_batch)
            _, mask_list, label_list, token_ids = zip(*samples)
            sent_vecs, mask_vecs, label_list, sent_lens = self.pad_data(token_ids,mask_list, label_list)
        else:
            if self.index == self.data_len:
                print('Testing Over!')
            #First get batches of testing data
            if self.data_len - self.index >= config.batch_size:
                #print('Testing Sample Index:', self.index)
                start = self.index
                end = start + config.batch_size
                samples = self.data_batch[start: end]
                self.index += config.batch_size
                tokens, mask_list, label_list, token_ids = zip(*samples)
                sent_vecs, mask_vecs, label_list, sent_lens = self.pad_data(token_ids,mask_list, label_list)
                #Sort the lengths, and change orders accordingly
                sent_lens, perm_idx = sent_lens.sort(0, descending=True)
                sent_vecs = sent_vecs[perm_idx]
                mask_vecs = mask_vecs[perm_idx]
                label_list = label_list[perm_idx]
            else:#Then generate testing data one by one
                samples =  self.data_batch[self.index] 
                _, mask_list, label_list, token_ids = zip(*[samples])
                sent_vecs, mask_vecs, label_list, sent_lens = self.pad_data(token_ids,mask_list, label_list)
                self.index += 1
        if with_text:
            yield sent_vecs, mask_vecs, label_list, sent_lens, tokens
        yield sent_vecs, mask_vecs, label_list, sent_lens

    def get_elmo_samples(self):
        '''
        Generate random samples for training process
        Generate samples for testing process
        sentences represented in Elmo
        '''
        if self.is_training:
            samples = self.generate_sample(self.data_batch)
            sent_vecs, mask_vecs, label_list, sent_lens = self.elmo_transform(samples)
            #Sort the lengths, and change orders accordingly
            sent_lens, perm_idx = sent_lens.sort(0, descending=True)
            sent_vecs = sent_vecs[perm_idx]
            mask_vecs = mask_vecs[perm_idx]
            label_list = label_list[perm_idx]
        else:
            if self.index == self.data_len:
                print('Testing Over!')
            #First get batches of testing data
            if self.data_len - self.index >= config.batch_size:
                #print('Testing Sample Index:', self.index)
                start = self.index
                end = start + config.batch_size
                samples = self.data_batch[start: end]
                self.index += config.batch_size
                sent_vecs, mask_vecs, label_list, sent_lens = self.elmo_transform(samples)
                #Sort the lengths, and change orders accordingly
                sent_lens, perm_idx = sent_lens.sort(0, descending=True)
                sent_vecs = sent_vecs[perm_idx]
                mask_vecs = mask_vecs[perm_idx]
                label_list = label_list[perm_idx]
            else:#Then generate testing data one by one
                samples =  self.data_batch[self.index] 
                sent_vecs, mask_vecs, label_list, sent_lens = self.elmo_transform([samples])
                self.index += 1
        yield sent_vecs, mask_vecs, label_list, sent_lens

        
    

# def read_data():
#     TRAIN_DATA_PATH = "data/2014/Restaurants_Train_v2.xml"
#     TEST_DATA_PATH = "data/2014/Restaurants_Test_Gold.xml"
#     dr = data_reader(config)
#     dr.read_raw_data(TRAIN_DATA_PATH)
#     dr.split_save_data('data/2014/training.pickle', 'data/2014/valid.pickle')
#     dr.read_raw_data(TEST_DATA_PATH)
#     dr.save_data('data/2014/testing.pickle')


# if __name__ == "__main__":
#     read_data()
