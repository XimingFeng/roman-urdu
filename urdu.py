from gensim.models import Word2Vec
import os
import pickle 
import re
import pandas as pd
import nltk
import numpy as np
import string

# def get_exist_word2vec():
#     word2vec_model = Word2Vec.load('word2vec.bin')
#     print('existing word2vec model loaded')
#     return word2vec_model
    
# def train_save_word2vec():
#     df = pd.read_csv('Roman Urdu DataSet.csv')
#     comment_list = []
#     pattern = re.compile("^[a-zA-Z]+$")
#     for idx, comments in df.iterrows():
#         words = nltk.word_tokenize(str(comments['Comment']))
#         for word in words:
#             if not pattern.match(word):
#                 words.remove(word)
#         comment_list.append(words)
#     print("comments tokenized, start to train Word2Vec model")
#     word2vec_model = Word2Vec(sentences=comment_list, min_count=1)
#     print("training completed")
#     word2vec_model.save('word2vec.bin')
#     print("model saved")
#     return word2vec_model

def get_unique_words():
    df = pd.read_csv('Roman Urdu DataSet.csv')
    word_list = ['unkwn']
    word2idx = {}
    pattern = re.compile("^[a-zA-Z]+$")
    unique_word_idx = 0
    for idx, comments in df.iterrows():
        words = nltk.word_tokenize(str(comments['Comment']))
        for word in words:
            word = word.lower()
            if pattern.match(word) and word not in word2idx:
                unique_word_idx += 1
                word_list.append(word)
                word2idx[word] = unique_word_idx 
                
    return word_list, word2idx

def get_processed_data(max_sent_len, word_list, word2idx):
    df = pd.read_csv('Roman Urdu DataSet.csv')
    total_entry_num = df.shape[0]
    print("Total data entry number: ", total_entry_num)
    X = np.zeros(shape=(total_entry_num, max_sent_len), dtype=np.int32)
    y = []
    pattern = re.compile("^[a-zA-Z]+$")
    unique_word_idx = 0
    for sent_idx, comments in df.iterrows():
        words = nltk.word_tokenize(str(comments['Comment']))
        word_idx = 0
        for word in words:
            word = word.lower()
            if pattern.match(word) and word in word2idx and word_idx < max_sent_len:
                X[sent_idx, word_idx] = word2idx[word]
                word_idx += 1
        sentiment_value = 0
        if str(comments['Sentiment']) == "Positive":
            sentiment_value = 1
        elif str(comments['Sentiment']) == "Neutral":
            sentiment_value = 2
        y.append(sentiment_value)
    y = np.array(y)
                
    return X, y

def get_train_batch(X_data, y_data, iteration, bs):
    start_idx = iteration * bs
    end_idx = start_idx + bs
    return X_data[start_idx: end_idx, :], y_data[start_idx: end_idx]

def get_random_batch(X_data, y_data, test_size):
    
    total_entry_num = X_data.shape[0]
    #print('total entry numver', total_entry_num)
    selected_idx = np.random.choice(total_entry_num, test_size, replace=False)
    return X_data[selected_idx, :], y_data[selected_idx]

# def build_rnn(sent_len, embed_size, num_classes, lstm_hidden_dim, batch_size, dense_dim, learn_rate):
#     tf.reset_default_graph()
    
#     X = tf.placeholder([None, sent_len], tf.int32)
#     y = tf.palceholder([None, num_classes])
#     embedding_lookup = tf.Variable(tf.random_uniform(shape=(sent_len, embed_size), minval=-1, maxval=1))
#     embed = tf.nn.embedding_lookup(params=embedding_lookup, ids=X)
#     lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_dim)
#     init_state = lstm_cell.zero_state(batch_size, tf.float32)
#     rnn_out, final_state = tf.nn.dynamic_rnn(lstm_cell, embed, initial_state=init_state)
#     dense_out = tf.contrib.layers.full_connected(rnn_out[:, -1], 
#                                                  num_outputs=dense_dim, 
#                                                  activation_fn=tf.sigmoid)
#     scores = tf.contrib.layers.full_connected(dense_out, 
#                                                  num_outputs=num_classes, 
#                                                  activation_fn=tf.sigmoid)
#     y_pred = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=scores)
#     loss = tf.reduce_mean(y_pred)
#     optmizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    
    
    
    
    
    
    
    
    
    