import re
import os
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pickle as pkl

class DataManager():
    def __init__(self):
        self.data = {}
    # Read data from data_path
    #  name       : string, name of data
    #  with_label : bool, read data with label or without label
    def add_data(self,name, data_path, with_label=True, skip_header=False):
        print ('read data from %s...'%data_path)
        X, Y = [], []
        with open(data_path,'r') as f:
            for line in f:
                if with_label:
                    lines = line.strip().split(' +++$+++ ')
                    X.append(lines[1])
                    Y.append(int(lines[0]))
                elif skip_header:
                    X.append(line.split(',', 1)[1])
                else:
                    X.append(line)

        if with_label:
            self.data[name] = [X,Y]
        elif skip_header:
            self.data[name] = [X[1:]]
        else:
            self.data[name] = [X]

    def preprocessing(self, filter=True, split=True):
        for key in self.data:
            print ('preprocessing %s'%key)
            for i in range(len(self.data[key][0])):
                st = self.data[key][0][i]
                st = re.sub(r'\' m(?= )', 'am', st)
                st = re.sub(r'(?<=he |it )\' s(?= |$)', 'is', st)
                st = re.sub(r'(?<=who |hat )\' s(?= |$)', 'is', st)
                st = re.sub(r'(?<=here |when )\' s(?= |$)', 'is', st)
                st = re.sub(r' \' s(?= |$)', '', st)
                st = re.sub(r'\' re(?= |$)', 'are', st)
                st = re.sub(r'\' ll(?= |$)', 'will', st)
                st = re.sub(r'\' ve(?= |$)', 'have', st)
                st = re.sub(r'can \' t', 'can not', st)
                st = re.sub(r'won \' t', 'will not', st)
                st = re.sub(r'n \' t(?= )', ' not', st)
                st = re.sub(r'\' d(?= )', 'would', st)
                st = re.sub(r'(?<= )u(?= )', 'you', st)
                st = re.sub(r'[^A-Za-z0-9 !?.,:\']', '', st)
                st = re.sub(r'(?<=(?P<a>[!?.,:\']))(?P=a){1,}', '', st)

                st = re.sub(r'(?<=\w{3})s(?= |$)', '', st)
                st = re.sub(r'(?<=\w{2})ing(?= |$)', '', st)
                
                st = re.sub(r'(?<= )[h|a]{2,}(?= |$)', 'ha', st)
                st = re.sub(r'(?<= )[h|e]{2,}(?= |$)', 'he', st)
                st = re.sub(r'(?<= )[x|o]{2,}(?= |$)', 'xo', st)
                st = re.sub(r'(?<= )[y|a]{2,}(?= |$)', 'ya', st)
                if filter:
                    #st = re.sub(r'(?<= (?P<a>\w))(?P=a){1,}', '', st)
                    #st = re.sub(r'(?<=(?P<a>\w))(?P=a){1,}(?= |$)', '', st)
                    st = re.sub(r'(?<=(?P<a>\w))(?P=a){1,}', '', st)
                    st = re.sub(r'(?<=(?P<a>\w{2}))(?P=a){1,}', '', st)
                    #st = re.sub(r'(?<= |^(?P<a>\w{2}))(?P=a){1,}', '', st)
                    #st = re.sub(r'(?<=(?P<a>\w){2})(?P=a){1,}(?= |$)', '', st)
                    #st = re.sub(r'(?<= |^(?P<a>\w{3}))(?P=a){1,}', '', st)
                    #st = re.sub(r'(?<=(?P<a>\w{3}))(?P=a){1,}(?= |$)', '', st)
                    '''
                    a = re.findall(r'(?<= (?P<a>[A-Za-z0-9]{2}))(?P=a){1,}', st)
                    if len(a):
                        print(self.data[key][0][i])
                        print(st)
                        print(i, ' ', a)
                    '''
                if split:
                    st = st.split()
                    #self.data[key][0][i] = re.split(r'[\n ]', self.data[key][0][i])
                #print(self.data[key][0][i])
                #print(st)
                self.data[key][0][i] = st


    # Build dictionary
    #  vocab_size : maximum number of word in yout dictionary
    def tokenize(self, vocab_size):
        print ('create new tokenizer')
        self.tokenizer = Tokenizer(num_words=vocab_size)
        for key in self.data:
            print ('tokenizing %s'%key)
            texts = self.data[key][0]
            self.tokenizer.fit_on_texts(texts)

    # Save tokenizer to specified path
    def save_tokenizer(self, path):
        print ('save tokenizer to %s'%path)
        pkl.dump(self.tokenizer, open(path, 'wb'))
            
    # Load tokenizer from specified path
    def load_tokenizer(self,path):
        print ('Load tokenizer from %s'%path)
        self.tokenizer = pkl.load(open(path, 'rb'))

    # Convert words in data to index and pad to equal size
    #  maxlen : max length after padding
    def to_sequence(self, maxlen, use_pretrain=False):
        self.maxlen = maxlen
        for key in self.data:
            print ('Converting %s to sequences'%key)
            if use_pretrain:
                sentences = self.data[key][0]
                tmp = []
                for i in range(len(sentences)):
                    st = []
                    for j in range(len(sentences[i])):
                        if sentences[i][j] in self.pre_model.wv.vocab:
                            #print(self.pre_model.wv.vocab[self.data[key][0][i][0]].index)
                            st.append(self.pre_model.wv.vocab[sentences[i][j]].index)
                    tmp.append(st)

            else:
                tmp = self.tokenizer.texts_to_sequences(self.data[key][0])
            self.data[key][0] = np.array(pad_sequences(tmp, maxlen=maxlen))
    
    # Convert texts in data to BOW feature
    def to_bow(self):
        for key in self.data:
            print ('Converting %s to tfidf'%key)
            self.data[key][0] = self.tokenizer.texts_to_matrix(self.data[key][0],mode='count')

    # Convert label to category type, call this function if use categorical loss
    def to_category(self):
        for key in self.data:
            if len(self.data[key]) == 2:
                self.data[key][1] = np.array(to_categorical(self.data[key][1]))

    def get_semi_data(self,name,label,threshold,loss_function) :
        # if th==0.3, will pick label>0.7 and label<0.3
        label = np.squeeze(label)
        index = (label>1-threshold) + (label<threshold)
        semi_X = self.data[name][0]
        semi_Y = np.greater(label, 0.5).astype(np.int32)
        if loss_function=='binary_crossentropy':
            return semi_X[index,:], semi_Y[index]
        elif loss_function=='categorical_crossentropy':
            return semi_X[index,:], to_categorical(semi_Y[index])
        else :
            raise Exception('Unknown loss function : %s'%loss_function)

    # get data by name
    def get_data(self,name='', all=False):
        if all:
            data = []
            for key in self.data:
                data += self.data[key][0]
            return data 
        return self.data[name]

    def load_word2vec(self, data_path):
        print ('load word2vec from %s...'%data_path)
        self.pre_model = Word2Vec.load(data_path)

    def load_embedding_matrix(self, data_path):
        self.embedding_matrix = np.load(data_path)

    def embedding_layer(self):
        return self.pre_model.wv.get_keras_embedding(False)

    def get_embedding_matrix(self):
        return self.embedding_matrix

    # split data to two part by a specified ratio
    #  name  : string, same as add_data
    #  ratio : float, ratio to split
    def split_data(self, name, ratio):
        data = self.data[name]
        X = data[0]
        Y = data[1]
        data_size = len(X)
        val_size = int(data_size * ratio)
        return (X[val_size:],Y[val_size:]),(X[:val_size],Y[:val_size])

if __name__=='__main__':
    dm = DataManager()
    dm.add_data('train', 'data/training_label.txt')
    dm.preprocessing(True, False)
    #dm.to_sequence(37, True)
    #print(dm.data['train'][0][1])
    #print(len(dm.data['train'][0]))

