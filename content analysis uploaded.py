# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 14:29:48 2022

@author: Yudi Chen
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime 
import pickle

import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from nltk.stem import 	WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
from collections import Counter

import matplotlib.pyplot as plt

from matplotlib import rcParams
params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal',
        'font.size':15
        }
rcParams.update(params)


with open('filtered_tweets.pickle', 'rb') as handle:
    tweets = pickle.load(handle)
    new_tweets = tweets.loc[tweets['tweet_type'] == 'original', :]
    retweets = tweets.loc[tweets['tweet_type'] != 'original', :]
    
    
def bot_user_remove(new_tweets):
    # remove the tweets that are possibly bots
    # 1. users with high posting frequency but low interactions
    # 2. users that have many duplicated tweets
    tweet_count = new_tweets.groupby(by='user_id').count()['hashtags'].sort_values(ascending=False)
    bot_ids = tweet_count.index[:3]
    for bot_id in bot_ids:
        new_tweets = new_tweets.loc[-(new_tweets['user_id'] == bot_id)]
    
    new_tweets.drop_duplicates(subset=['full_text', ], keep='first', inplace=True)
    
    return new_tweets


def ratio_calculate(new_tweets):
    '''
    Calculate the distribution of Tweeting activities
    Pareto principles are included 20% - 80%
    '''
    new_tweets['retweet_count'] = new_tweets['retweet_count'] + 1
    retweet_count = new_tweets['retweet_count'].sort_values(ascending=False)
    cumsum_ratio = retweet_count.cumsum() / retweet_count.sum()
    plt.figure()
    xticks = np.array(range(1, cumsum_ratio.shape[0] + 1))
    xticks = xticks / cumsum_ratio.shape[0]
    plt.plot(xticks, cumsum_ratio.values, '-b')
    plt.plot([0, 1], [0.8, 0.8], '--r')
    plt.scatter([xticks[1072], ], [cumsum_ratio.iloc[1072], ], marker='o', s=100, 
                c='r')
    plt.text(xticks[1072], cumsum_ratio.iloc[1072] - 0.075, 
             '(' + str(xticks[1072])[2:4] + '%, ' + str(cumsum_ratio.iloc[1072])[2:4] + '%)')
    plt.xlabel('Percentage of the Tweets')
    plt.ylabel('Cumulative ratio')
    plt.xlim(0, 1)
    plt.gcf().tight_layout()
    plt.gcf().savefig('pareto_principle.tif', dpi=300)
    
    print('The number of Tweets reaching 80% activity is {}'.format(sum(cumsum_ratio < 0.8)))
    
    return cumsum_ratio


# new_tweets = bot_user_remove(new_tweets)
# cumsum_ratio = ratio_calculate(new_tweets)


def tweeting_stats():
    # import labeled tweets
    tweets = pd.read_csv('majority_tweets_1072.csv')
    print('# of labeled tweets is {}'.format(tweets.shape[0]))
    tweets_interest = tweets.loc[tweets['Category'] != 'others']
    print('# of interest tweets is {}'.format(tweets_interest.shape[0]))
    
    cats = tweets_interest['Category'].unique()
    retweet_lst = []
    count_lst = []
    for cat in cats:
        cat_tweets = tweets_interest.loc[tweets_interest['Category'] == cat]
        cat_groupby = cat_tweets.groupby(by='Brand image')
        count_lst.append(cat_groupby.count()['user_id'])
        retweet_lst.append(cat_groupby.sum()['retweet_count'])
    
    count_df = pd.concat(count_lst, axis=1).T
    retweet_df = pd.concat(retweet_lst, axis=1).T
    count_df.set_index(cats, inplace=True)
    retweet_df.set_index(cats, inplace=True)
    ratio_df = count_df / retweet_df
    
    return count_df, retweet_df, ratio_df
    
    
    
# count_df, retweet_df, ratio_df = tweeting_stats()

# with open('new_tweets.pickle', 'wb') as handle:
#     pickle.dump(new_tweets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('retweets.pickle', 'wb') as handle:
#     pickle.dump(retweets, handle, protocol=pickle.HIGHEST_PROTOCOL)









