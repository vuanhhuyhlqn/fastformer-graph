import pandas as pd
import pickle

def build_news_id_map(data_path='data/MINDsmall_train/'):
    news_df = pd.read_csv(data_path + 'news.tsv', sep='\t', header=None)
    news_ids = news_df[0].unique().tolist()

    news_id_map = {news_id: i + 1 for i, news_id in enumerate(news_ids)}
    news_id_map['PAD'] = 0

    with open(data_path + 'news_id_map.pkl', 'wb') as f:
        pickle.dump(news_id_map, f)

def build_user_id_map(data_path='data/MINDsmall_train/'):
    behavior_df = pd.read_csv(data_path + 'behavior.tsv', sep='\t', header=None)

    unique_user = sorted(behavior_df[1].unique().tolist())

    user_id_map = {uid: i + 1 for i, uid in enumerate(unique_user)}
    user_id_map['UNK_USER'] = 0

    with open(data_path + 'user_id_map.pkl', 'wb') as f:
        pickle.dump(user_id_map, f)
