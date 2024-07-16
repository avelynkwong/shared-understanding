import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
import nltk
from nltk.corpus import stopwords
import mysql.connector
import string
import os
import datetime
import numpy as np
import json
import re
import enchant
import spacy
import math
import time
import matplotlib.dates as mdates
import os.path
from gensim import corpora
from gensim.models import LsiModel, LogEntropyModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from gensim import matutils
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine
from aggregation import message_aggregation

def token_stem_stop(docs):
    # initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = set(stopwords.words('english'))
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # list for tokenized documents in loop
    texts = []
    # loop through document list
    doc_set = docs.tolist() 
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens)
    return texts

def prepare_corpus(processed_docs):
    """
    Input  : clean document
    Purpose : create term dictionary of our corpus and convert list of documents (corpus) into a word-doc matrix
    Output : term dictionary and word-doc matrix
    """
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(processed_docs)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_docs] #doc2bow function returns tuples of type (word id, frequency)
    # generate LDA model
    return dictionary, doc_term_matrix


# trains different LSA models with varying number of topics and outputs the best model
def train_LSA_models(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
    """
    Parameters:
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        stop: maximum number of topics
    Outputs:
        best_model: the LSA model with the highest coherence score
    """
    best_coherence = 0
    best_model = None
    for num_topics in range(start, stop, step):
        # train LSA model
        model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary)
        coherencemodel = CoherenceModel(
            model=model, texts=doc_clean, dictionary=dictionary, coherence="c_v"
        )
        model_coherence = coherencemodel.get_coherence()
        if model_coherence > best_coherence:
            best_model = model
            best_coherence = model_coherence
    return best_model


def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


# Normalize the bow representation of the document
def get_normalized_bow(doc, dictionary):
    bow = dictionary.doc2bow(doc)
    bow_dense = np.zeros(len(dictionary))
    for idx, value in bow:
        bow_dense[idx] = value
    normalized_bow = normalize_data(bow_dense.reshape(1, -1))
    return list(zip(range(len(normalized_bow[0])), normalized_bow[0]))


# Apply logent_transformation on normalized data
def apply_transformation(doc, dictionary, logent_transformation):
    normalized_bow = get_normalized_bow(doc, dictionary)
    return logent_transformation[normalized_bow]


def build_model(matrix, dictionary, df_processed, stop_div, step):

    # log entropy transformation
    logent_transformation = LogEntropyModel(matrix, dictionary)
    logent_corpus = logent_transformation[matrix]
    dense_logent_matrix = matutils.corpus2dense(
        logent_corpus, num_terms=len(dictionary)
    ).T
    filtered_corpus = matutils.Dense2Corpus(dense_logent_matrix.T)

    # Create a new dictionary with the filtered terms
    filtered_dictionary = dictionary
    best_model = train_LSA_models(
        filtered_dictionary,
        filtered_corpus,
        df_processed,
        stop=len(df_processed) // stop_div,
        step=step,
    )
    return best_model, logent_transformation, logent_corpus

def memory_coherence(df, best_model, logent_transformation, dictionary): 
    list_output = [] 
    for channel in df['channel_id'].unique():
        for time in  df['timestamp'].unique():
            
            filtered_df = df[(df['channel_id']==channel) & (df['timestamp'] == time)]
            if len(filtered_df)==0:
                continue
    
            for i, row in filtered_df.iterrows(): 
                document_distribution = best_model[logent_transformation[(dictionary.doc2bow(row['processed_for_LSA']))]]
                print(document_distribution)
            
                list_output.append({
                    'channel': channel,
                    'channel_name': filtered_df['channel_name'].iloc[0],
                    'timestamp': time,
                    'num_users': len(filtered_df['id_user'].unique()),
                    'user': row['id_user'],
                    'matrix': document_distribution
                })

    df_output = pd.DataFrame(list_output)

    #calculate the final group centroid
    total_matrix = [item for sublist in df_output['matrix'] for item in sublist]
    print(total_matrix)
    if len(total_matrix) >0:
        if type(total_matrix[0]) is tuple:
            total_matrix = [total_matrix]total_df = matrix_to_df(total_matrix)
    print(total_df)
    group_total_centroid = total_df.mean(axis=1)

    #now go through and calculate the running coherence 
    list_output2 = []

    for channel in df_output['channel'].unique():
        #print(channel)
        for time in  df_output['timestamp'].unique():
            #print(time)

            #calculate group coherence 
            df_filtered = df_output[(df_output['channel']==channel) & (df['timestamp'] <= time)]

            #print(len(df_filtered))

            if len(df_filtered)==0:
                print("empty df v2")
                continue

            num_users = len(df_filtered['user'].unique())
            num_user_list = [num_users]*(num_users+1)

            channel_list = [channel]*(num_users+1)
            time_list = [time]*(num_users+1)
            channel_name_list = [df_filtered['channel_name'].iloc[0]]*(num_users+1)
            #print(df_filtered['matrix'])
            #append all the matrixes together
            channel_time_matrix = [item for sublist in df_filtered['matrix'] for item in sublist]
            #print(channel_time_matrix)
            if len(channel_time_matrix) >0:
                if type(channel_time_matrix[0]) is tuple:
                    channel_time_matrix = [channel_time_matrix]
            #print(channel_time_matrix)
            #if channel_time_matrix[0, 0] == 0
            channel_time_df = matrix_to_df(channel_time_matrix)
            group_centroid = channel_time_df.mean(axis=1)
            group_centroid_distance = cosine(group_centroid, group_total_centroid)

            users = []
            user_dists = []
            #print("getting into user loop")
            for user in df_filtered['user'].unique():
                users.append(user)
                df_filtered_user = df_filtered[df_filtered['user']==user]
                channel_time_user_matrix = [item for sublist in df_filtered_user['matrix'] for item in sublist]
                #print(channel_time_user_matrix)
                if len(channel_time_user_matrix) >0:
                    if type(channel_time_user_matrix[0]) is tuple:
                        channel_time_user_matrix = [channel_time_user_matrix]

                channel_time_user_df = matrix_to_df(channel_time_user_matrix)
                user_cent = channel_time_user_df.mean(axis=1)
                user_distance = cosine(user_cent, group_total_centroid)
        

                user_dists.append(user_distance)
            users.append("Group")
            user_normsdistsnd(group_centroid_distance)
            stdev_list = [np.std(user_dists)]*(num_users+1)


             # Flatten the dictionary
            for i in range(len(users)):
                list_output2.append({
                    'channel': channel_list[i],
                    'channel_name': channel_name_list[i],
                    'timestamp': time_list[i],
                    'num_users': num_user_list[i],
                    'user': users[i],
                    'coherence': user_dists[i],
                    'std': stdev_list[i]
                })

    df_output2 = pd.DataFrame(list_output2)

    return df_output2

def LSA(preprocessed_df, agg_type, stop_div, step, memory):
    """
    Inputs:
        preprocessed_df: dataframe that has already be preprocessed with general_preprocessing (remove non dict words, etc.)'
        agg_type: how messages are grouped into documents
        
    """

    agg_df = message_aggregation(agg_type, preprocessed_df)
    # tokenize the messages
    lsa_processed_docs = token_stem_stop(agg_df['text'])
    agg_df['processed_for_LSA'] = lsa_processed_docs
    # create dictionary of words and word-document matrix
    dictionary, matrix = prepare_corpus(lsa_processed_docs)

    model_df, logent_transformation, logent_corpus = build_model(matrix, dictionary, df_processed, stop_div, step)

    max_coherence_index = max(model_df['coherence'])

    # Find all indices of the maximum value
    max_indices = [index for index, value in enumerate(model_df['coherence']) if value == max_coherence_index]

    best_model = model_df['model'].iloc[max_indices[0]]
    print("best model found")

    if memory ==1:
        df_output = memory_coherence(test_output, best_model, logent_transformation, dictionary)
    elif memory ==0: 
        df_output = no_memory_coherence(test_output, best_model, logent_transformation, dictionary)

    print("coherence computed")


    

    LSA_viz(df_output, agg_type)


