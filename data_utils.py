import pandas as pd

def build_news_id_map(data_path='data/MINDsmall_train/'):
    news_df = pd.read_csv(data_path + 'news.tsv', sep='\t', header=None)
    news_ids = news_df[0].unique().tolist()

    news_id_map = {news_id: i + 1 for i, news_id in enumerate(news_ids)}
    news_id_map['PAD'] = 0