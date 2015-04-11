# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 14:20:12 2015

@author: jfherbst
"""# -*- coding: utf-8 -*-

#!/usr/local/bin/python
###################################################
#
# soupstart2.py
# Goal: process text from SOUP proposals 
# By Jeffrey Herbstman, 4-5-2015
# 
# Inputs:
#
# Outputs:
# single file with name provided as input (see above)
#
#Options to possibly add:
#
#
###################################################
import os
import pandas as pd
import re
import wordcloud
import matplotlib.pyplot as plt
import string
import sys, nltk, collections, sklearn.feature_extraction.text, time
from collections import Counter
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer
from IPython import embed
from pprint import pprint
import heapq


#============================
def import_mgmt():
    #winners = pd.read_csv('win2.csv')
    #pd.value_counts(winners['Year'])
    #winners['Number of People'].describe()
    #winners.describe()
    
    props = pd.read_csv('prop_c2.csv')
    #shape(props)
    props.rename(columns={'Project Summary': 'Summary', \
    'Why does this project matter to the community?':'Why', \
    'How will you use SOUP grant funding towards the realization of your project?':'How'}, inplace=True)
    
    return props

#============================
def preproc(props, lem=True):
    props['Summary'] = props['Summary'].str.lower()
    props['Why'] = props['Why'].str.lower()
    props['How'] = props['How'].str.lower()
    props['text']=props['Summary']+props['Why']+props['How']
    props = props[pd.notnull(props['text'])]
    
    return props
    
#============================
    
def tokensizer(textsingle):
    

    textsingle = textsingle.encode('utf-8').translate(None, string.punctuation)
    tokens = nltk.word_tokenize(textsingle)   
    wnl = nltk.WordNetLemmatizer()
    
    mappedlem = map(wnl.lemmatize,tokens)
    
    
    return mappedlem
    
#============================
def clustertext(wordlist, clusters=8):
    global tfidf_model
    global vectorizer    
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    ignore = set(nltk.corpus.stopwords.words('english'))
    vectorizer = TfidfVectorizer(tokenizer=tokensizer,
                                 stop_words=ignore,
                                 #max_df=0.5,
                                 #min_df=0.1,
                                 lowercase=True)
    
 
    tfidf_model = vectorizer.fit_transform(wordlist)
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model) 
    
    clustering = collections.defaultdict(list)
 
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
    return clustering, tfidf_model, vectorizer
    
#============================
def topfeaturenames(vectorizer, tfidf_model):
    
    featurenames = vectorizer.get_feature_names()
    dictlist = []
    
    
    for idx,response in enumerate(tfidf_model):
        featlist = dict()
        for col in response.nonzero()[1]:
            featlist[featurenames[col]] = response[0, col]
        dictlist.append(heapq.nlargest(20, featlist, key=featlist.get))
    return dictlist       
#============================
def wordclouding(clustex,dictlist):
    
    cloudlist = []
    for clusternum in clustex:
        clustcommon=[]
        for obsnum in clustex[clusternum]:
            clustcommon +=dictlist[obsnum]
        c = Counter(clustcommon)
        ctrunc = c.most_common(30)
        cloudlist.append(ctrunc)
        clustcloud = wordcloud.WordCloud()
        clustcloud.fit_words(ctrunc)
        clustcloud.words_ = ctrunc
    return cloudlist

        
    
#============================
def main():
        
    texts = import_mgmt() 
    texts = preproc(texts)
    
    #print(shape(texts))
    onecol = texts['text']
    clusters, model, vec = clustertext(onecol)
    #pprint(dict(clusters))
    clustex = clusters
    dictlist = topfeaturenames(vec, model) 
    cloudlist = wordclouding(clustex, dictlist)
    
    #here's code for word clouds
    
#        clustcloud = wordcloud.WordCloud()
#        clustcloud.fit_words(cloudlist[5])
#        clustcloud.words_ = cloudlist[5]
#        plt.imshow(clustcloud)
#        plt.axis("off")
#        plt.show()
    
if __name__ == "__main__":
    main()