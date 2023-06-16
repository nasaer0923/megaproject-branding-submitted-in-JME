# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:04:16 2022

@author: Yudi Chen
"""

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import numpy as np

colors = ['gray', '#ffe6e6', '#fcf8c1', '#defcce', '#e9dbf9', '#ceeef9']

def tweet_import():    
    with open('retweets.pickle', 'rb') as handle:
        retweets = pickle.load(handle)
        
    with open('new_tweets.pickle', 'rb') as handle:
        new_tweets = pickle.load(handle)
    
    labeled_tweets = pd.read_csv('majority_tweets_1072.csv')
    labeled_tweets.dropna(axis=0, how='any', subset=['Brand image', ], inplace=True)
    return new_tweets, labeled_tweets, retweets


def lookup_tweet_id(new_tweets, labeled_tweets):
    '''identifiy the tweet ID for each labeled tweets from the new tweets
    three keys:
        full_text
        created_at
        follower_count
    '''
    new_tweets['created_at'] = new_tweets['created_at'].apply(pd.to_datetime)
    labeled_tweets['created_at'] = labeled_tweets['created_at'].apply(pd.to_datetime)
    
    user_ids = []
    tweet_ids = []
    for ii in range(labeled_tweets.shape[0]):
        full_text = labeled_tweets.iloc[ii]['full_text']
        created_at = labeled_tweets.iloc[ii]['created_at']
        follower_count = labeled_tweets.iloc[ii]['follower_count']
        text_tag = (new_tweets['full_text'] == full_text)
        time_tag = (new_tweets['created_at'] == created_at)
        follower_tag = (new_tweets['follower_count'] == follower_count)
        sum_tag = [i + j + k for i, j, k in zip(text_tag, time_tag, follower_tag)]
        try:
            idx = sum_tag.index(3)
            user_ids.append(new_tweets.iloc[idx]['user_id'])
            tweet_ids.append(str(new_tweets.iloc[idx]['tweet_id']))
        except ValueError:
            user_ids.append(None)
            tweet_ids.append(None)
            print('ID is not found for {}'.format(full_text))
    
    labeled_tweets['tweet_id'] = tweet_ids
    labeled_tweets['user_id'] = user_ids
    
    return labeled_tweets


def retweet_filtering(retweets, labeled_tweets):
    '''Only keep the retweets that are related to the labeled tweets
    '''
    labeled_tweet_ids = labeled_tweets['tweet_id'].to_list()
    idxs = []
    for ii in range(retweets.shape[0]):
        retweet_id = retweets.iloc[ii]['related_tweet_id']
        if str(retweet_id) in labeled_tweet_ids:
            idxs.append(ii)
            
    filtered_retweets = retweets.iloc[idxs]
    
    return filtered_retweets


def get_graph_attr(G):
    '''
    Return a set of attributes of the graph, including:
        the number of edges
        the number of nodes
        the density of the graph
    '''
    attributes = {}
    attributes['num_node'] = len(G.nodes())
    attributes['num_edge'] = len(G.edges())
    if not nx.is_directed(G):
        attributes['density'] = G.density()
    
    return attributes

import scipy.stats as ss
def get_top_nodes(G, k, plot_flag=True):
    '''
    Get the top nodes with the highest measure
    Parameters
    ----------
    G : Graph
    measure : Categorical
        Degree for undirecrted Graph, in_degree, out_degree for directed Graph.
        Betweenness
        Eigenvalue

    Returns
    -------
    top_nodes : DataFrame

    '''
    if nx.is_directed(G):
        top_nodes = pd.DataFrame(index=range(1, k+1), columns=['in_degree_nodes', 'in_degree_values', 
                                                               'out_degree_nodes', 'out_degree_values'])
        in_degree = pd.Series(dict(G.in_degree(G.nodes()))).sort_values(ascending=False)
        out_degree = pd.Series(dict(G.out_degree(G.nodes()))).sort_values(ascending=False)
        top_nodes['in_degree_nodes'] = in_degree.index[:k]
        top_nodes['in_degree_values'] = in_degree.iloc[:k].values
        top_nodes['out_degree_nodes'] = out_degree.index[:k]
        top_nodes['out_degree_values'] = out_degree.iloc[:k].values
        nx.set_node_attributes(G, in_degree.to_dict(), 'in_degree')
        nx.set_node_attributes(G, out_degree.to_dict(), 'out_degree')
    else:
        top_nodes = pd.DataFrame(index=range(1, k+1), columns=['degree_nodes', 'degree_values'])
        degree = pd.Series(dict(G.degree(G.nodes()))).sort_values(ascending=False)
        top_nodes['degree_nodes'] = degree.index[:k]
        top_nodes['degree_values'] = degree.iloc[:k].values
        nx.set_node_attributes(G, degree.to_dict(), 'degree')
        
    betweenness = pd.Series(nx.betweenness_centrality(G)).sort_values(ascending=False)
    top_nodes['betweenness_nodes'] = betweenness.index[:k]
    top_nodes['betweenness_values'] = betweenness.iloc[:k].values
    nx.set_node_attributes(G, betweenness.to_dict(), 'betweeenness')
    
    if plot_flag:
        if nx.is_directed(G):
            fig, axes = plt.subplots(nrows=1, ncols=3)
            axes = axes.flatten()
            axes[0].hist(in_degree.values, bins=20, density=True)
            axes[0].set_xlabel('In-degree value')
            axes[0].set_ylabel('Frequency')
            
            axes[1].hist(out_degree.values, bins=20, density=True)
            axes[1].set_xlabel('Out-degree value')
            axes[1].set_ylabel('Frequency')
            
            axes[2].hist(betweenness.values, bins=100, density=True)
            axes[2].set_xlabel('Betweenness')
            axes[2].set_ylabel('Probability')
        else:
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes = axes.flatten()
            axes[0].hist(degree.values, bins=20, density=True)
            axes[0].set_xlabel('Degree value')
            axes[0].set_ylabel('Frequency')
            
            axes[1].hist(betweenness.values, bins=20, density=True)
            axes[1].set_xlabel('Betweenness value')
            axes[1].set_ylabel('Frequency')
            fig.tight_layout()
    
    return top_nodes


def degree_plot(G):
    '''Plot the histgram of degree distributions
    '''
    degree = pd.Series(dict(G.degree(G.nodes()))).sort_values(ascending=False)
    plt.figure()
    # params = ss.expon.fit(degree.values)
    # rx = np.linspace(0, max(degree.values), 100)
    # rp = ss.expon.pdf(rx, *params)
    plt.hist(degree.values, bins=563, density=True, alpha=0.6, color='blue', 
             # log=True
             )
    # plt.plot(rx, rp, '--r')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.tight_layout()


def graph_building(retweets, directed=True):
    '''Create the Graph based on the information in retweets
    '''
    edge_lst = []
    for ii in range(retweets.shape[0]):
        source_id = retweets['related_user_id'].iloc[ii] 
        target_id = retweets['user_id'].iloc[ii]
        if source_id != target_id:
            edge_lst.append([source_id, target_id])
    
    if directed:
        G = nx.DiGraph(edge_lst)
    else:
        G = nx.Graph(edge_lst)
    
    return G


def get_top_user(user_ids, labeled_tweets):
    
    top_user_tweets = pd.DataFrame(columns=labeled_tweets.columns)
    for user_id in user_ids:
        top_user_tweets = top_user_tweets.append(labeled_tweets.loc[labeled_tweets['user_id'] == user_id])
    
    return top_user_tweets


def community_ratio_calc(user_tweets):
    '''Calculate the percentage of a tweet in a community
    '''
    user_ids = user_tweets['user_id'].unique()
    ratio_df = pd.DataFrame(columns=['user_id', 'percentage', 'category', 'brand image'])
    for user_id in user_ids:
        tmp_ratio_df = pd.DataFrame(columns=['user_id', 'percentage', 'category', 'brand image'])
        user_df = user_tweets.loc[user_tweets['user_id'] == user_id][['retweet_count', 'Category', 'Brand image']]
        ratios = user_df['retweet_count'].values / user_df['retweet_count'].sum()
        tmp_ratio_df['category'] = user_df['Category']
        tmp_ratio_df['brand image'] = user_df['Brand image']
        tmp_ratio_df['user_id'] = [user_id, ] * user_df.shape[0]
        tmp_ratio_df['percentage'] = ratios
        ratio_df = ratio_df.append(tmp_ratio_df)
        print(user_id)
        print(user_df.shape[0])
        
    return ratio_df


def get_tweeting_stats(labeled_tweets):
    
    tweet_count = labeled_tweets.groupby(['Category', 'Brand image']).size().to_frame()
    tweet_count.reset_index(level=1, inplace=True)
    tweet_count = pd.pivot_table(tweet_count, values=0, index=tweet_count.index, columns='Brand image')
    tweet_count.drop(labels=['others', ], axis=0, inplace=True)
    
    tweet_ratio = labeled_tweets.groupby(['Category', 'Brand image']).sum()['retweet_count'].to_frame()
    tweet_ratio.reset_index(level=1, inplace=True)
    tweet_ratio = pd.pivot_table(tweet_ratio, values='retweet_count', index=tweet_ratio.index, columns='Brand image')
    tweet_ratio.drop(labels=['others', ], axis=0, inplace=True)
    tweet_ratio = tweet_count / tweet_ratio.values

    return tweet_count, tweet_ratio


from scipy.stats import entropy
def entropy_calc(retweet_counts):
    retweet_counts = retweet_counts + 1
    retweet_counts = retweet_counts / retweet_counts.sum()
    print(retweet_counts)
    norm_entropy = entropy(retweet_counts, base=2) / np.log2(retweet_counts.shape[0])
    
    return norm_entropy


def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return 1 - diffsum / (len(x)**2 * np.mean(x))



def get_tweeting_variesty(labeled_tweets):
    '''Derive the Tweeting variety for each pair of content and brand image
    '''
    contents = list(labeled_tweets['Category'].unique())
    images = list(labeled_tweets['Brand image'].unique())
    tv_df = pd.DataFrame(index=contents, columns=images)
    count_df = pd.DataFrame(index=contents, columns=images)
    for content in contents:
        for image in images:
            idx = (labeled_tweets['Category'] == content) & (labeled_tweets['Brand image'] == image)
            pair_tweets = labeled_tweets.loc[idx]
            count_df.loc[content, image] = sum(idx)
            tv_df.loc[content, image] = gini_coefficient(pair_tweets['retweet_count'].values)
    
    return tv_df, count_df


def pattern_analysis(G):
    '''Calculate the l metric used for indicating diffusion patterns
    '''
    outdegree = pd.Series(dict(G.out_degree(G.nodes()))).sort_values(ascending=False)
    node_size_lst = []
    edge_size_lst = []
    l_metric_lst = []
    for component in nx.weakly_connected_components(G):
        component = list(component)
        gsub = G.subgraph(component)
        node_size_lst.append(len(gsub.nodes))
        edge_size_lst.append(len(gsub.edges))
        leaf_nodes = [node for node in gsub.nodes if outdegree.loc[node] == 0]
        l_metric_lst.append(len(leaf_nodes) / len(component))
        
    df = pd.DataFrame(data=[l_metric_lst, node_size_lst, edge_size_lst],
                      index=['L metric', 'node size', 'edge size']).T
    
    return df


# load the data
new_tweets, labeled_tweets, retweets = tweet_import()

# find the user id and tweet id for each labeled tweet
labeled_tweets = lookup_tweet_id(new_tweets, labeled_tweets)

# only keep the retweets that are related to one of the labeled tweets
# this is critical as it provides tweet contents posted by opinion leaders
filtered_retweets = retweet_filtering(retweets, labeled_tweets)

#build the graph based on the relations between retweet users and original users
G = graph_building(filtered_retweets, directed=True)

pattern_info = pattern_analysis(G)
# find the top nodes based on in_degree, out_degree, and betweeness
topnodes = get_top_nodes(G, k=20)

# user_ids = topnodes['out_degree_nodes']
# outdegree_user_tweets = get_top_user(user_ids, labeled_tweets)
# ratio_df = community_ratio_calc(outdegree_user_tweets)

tv, count = get_tweeting_variesty(labeled_tweets)



