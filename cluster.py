import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import umap
import umap.plot

file = 'data/full_layer_1_lda.csv'
song_lbls = ['title', 'artist', 'lastfm_id', 'genius_id']
embedding_lbls = [str(i) for i in range(0, 768)]
other_lbls = ['feature_acousticness', 'feature_mode', 'feature_energy', 'feature_instrumentalness', 'feature_liveness', 'feature_valence', 'feature_loudness', 'feature_speechiness', 'feature_tempo', 'feature_key', 'feature_time_signature']
result_lbls = ['tag',] + [str(i) for i in np.arange(0.1, 7.1, 1)]


def preprocessed_df():
    '''
    Categorical data that needs to be split:
        1. feature_key *12 total
        2. feature_time_signature * 4 total

    Data that needs to be normalized:
        1. all embedding layers
        2. feature_loudness
        3. feature_tempo
        4. feature_speechiness?? Don't do this one for now

    '''
    df = pd.read_csv(file, usecols=song_lbls + embedding_lbls + other_lbls)[song_lbls + embedding_lbls + other_lbls]
    
    # one hot encoding on time signature and key
    t = 'feature_time_signature'
    k = 'feature_key'

    time_sig = pd.get_dummies(df[t], prefix=t)
    key = pd.get_dummies(df[k], prefix=k)

    df = pd.concat([df, time_sig, key], axis=1)

    # drop the columns since they are split
    df.drop([t, k], axis=1, inplace=True)

    # normalize embeddings, loudness, and tempo
    cols_to_norm = embedding_lbls + ['feature_loudness', 'feature_tempo']
    df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    return df


def reduce_dimensions(df):
    # keep high dimensional structure in low dimension
    # orignal embeddings has 767 features
    # for i in [0.5,0.6,0.7, 0.8]:
    # for i in [2,5,10,50,100]:
    #     print(i)
    bert_umap = umap.UMAP(n_neighbors=3,
                     n_components=5,
                     min_dist = 0.65,
                     metric='euclidean').fit(df)
    umap_embeddings = bert_umap.transform(df)
    
    return bert_umap, umap_embeddings


def dbscan(embeddings, eps=3, min_samples=2):
    cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    return cluster.labels_


def show_prediction(df, clusters):
    # Prepare data
    bert_umap = umap.UMAP(n_neighbors=3,
                 n_components=2,
                 min_dist = 0.65,
                 metric='euclidean').fit_transform(df)
    result = pd.DataFrame(bert_umap, columns=['x', 'y'])
    
    result['labels'] = clusters   # grab labels from clustering

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=100)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=100, cmap='viridis')

    #single point
    plt.plot(result.loc[0,'x'],result.loc[0,'y'], 'rx', mew=8, ms=18) 
    plt.show()

def get_precision(clusters):
    #takes clusters and get correct cluster to total set
    correct = [x for x in clusters[1:] if x == clusters[0]]
    #if max(clusters) == 0:
    #    return 0
    return len(correct)/(len(clusters)-1)
    
