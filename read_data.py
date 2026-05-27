import numpy as np
import pandas as pd

def load_data(dataset):
    # Load Data
    dataset_dir = "data/" + dataset + "/" + "ratings.dat"
    if dataset == "ml-1m":
        rating = pd.read_csv(dataset_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif dataset in ["100k", 'lastfm-2k', 'supplier']:
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
        rating = rating.sort_values(by='uid', ascending=True)
    elif dataset == 'hetrec':
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'])
    elif dataset == "filmtrust":
        rating = pd.read_csv(dataset_dir, sep="\t", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif dataset in ["beauty" , 'ku', 'card' , 'video']:
        dataset_dir = "data/" + dataset + "/" + "ratings.csv"
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                             engine='python')
        rating = rating.sort_values(by='uid', ascending=True)
    else:
        pass

    # Reindex
    user_id = rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    rating = pd.merge(rating, user_id, on=['uid'], how='left')
    item_id = rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    rating = pd.merge(rating, item_id, on=['mid'], how='left')
    rating = rating[['userId', 'itemId', 'rating', 'timestamp']]
    rating["rating"] = rating["rating"].astype("Int32")
    rating["timestamp"] = rating["timestamp"].astype("Int64")
    return rating