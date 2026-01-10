import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import torch

def build_news_id_map(data_path='data/MINDsmall_train/'):
	news_df = pd.read_csv(data_path + 'news.tsv', sep='\t', header=None)
	news_ids = news_df[0].unique().tolist()

	news_id_map = {news_id: i + 1 for i, news_id in enumerate(news_ids)}
	news_id_map['PAD'] = 0

	with open(data_path + 'news_id_map.pkl', 'wb') as f:
		pickle.dump(news_id_map, f)

def build_user_id_map(data_path='data/MINDsmall_train/'):
	behavior_df = pd.read_csv(data_path + 'behaviors.tsv', sep='\t', header=None)

	unique_user = sorted(behavior_df[1].unique().tolist())

	user_id_map = {uid: i + 1 for i, uid in enumerate(unique_user)}
	user_id_map['UNK_USER'] = 0

	with open(data_path + 'user_id_map.pkl', 'wb') as f:
		pickle.dump(user_id_map, f)

def build_news_token(news_id_map, data_path, max_len=64):
	tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
	news_df = pd.read_csv(data_path + 'news.tsv', sep='\t', header=None)
	raw_data = {
		row[0]: str(row[3]) + " " + tokenizer.sep_token + " " + str(row[4]) 
		for _, row in news_df.iterrows()
	}   

	num_news = len(news_id_map)
	sorted_data = np.zeros((num_news, max_len), dtype=np.int32)
	
	for nid, idx in tqdm(news_id_map.items(), desc="Tokenizing"):
		if nid == 'PADD':
			continue

		text = raw_data.get(nid, "")

		tokens = tokenizer.encode(
			text, 
			add_special_tokens=True,
			max_length=max_len,
			padding='max_length',
			truncation=True
		)

		sorted_data[idx] = tokens
	np.save(data_path + 'news_token.npy', sorted_data)

def build_edge_index(behavior_path, news_id_map, save_path='edge_index.pt'):
	df = pd.read_csv(behavior_path, sep='\t', header=None, usecols=[3])
	
	edges = set()
	
	for history in tqdm(df[3].dropna()):
		news_list = history.split()
		if len(news_list) < 2:
			continue
			
		indices = [news_id_map[nid] for nid in news_list if nid in news_id_map]
		
		for i in range(len(indices) - 1):
			edges.add((indices[i], indices[i+1]))
			
	edge_list = np.array(list(edges)).T 
	edge_index = torch.tensor(edge_list, dtype=torch.long)
	
	torch.save(edge_index, save_path)
	
	return edge_index



def build_behaviors(behaviors_path, user_id_map, news_id_map, save_path, max_hist=50):
	df = pd.read_csv(behaviors_path, sep='\t', header=None, usecols=[1, 3, 4])
	df.columns = ['user_id', 'history', 'impressions']
	user_indices = df['user_id'].map(user_id_map).fillna(0).astype(np.int32).values

	def process_history(h_str):
		if pd.isna(h_str) or h_str == '':
			return np.zeros(max_hist, dtype=np.int32)
		h_list = [news_id_map.get(nid, 0) for nid in h_str.split()]
		if len(h_list) >= max_hist:
			return np.array(h_list[-max_hist:], dtype=np.int32)
		return np.pad(h_list, (max_hist - len(h_list), 0), 'constant')
	
	histories = np.array([process_history(h) for h in df['history']], dtype=np.int32)

	converted_data = []

	for u_idx, hist, imp in tqdm(zip(user_indices, histories, df['impressions'])):
		pos_ids = []
		neg_ids = []
		for item in imp.split():
			nid, label = item.split('-')
			n_idx = news_id_map.get(nid, 0)
			if label == '1':
				pos_ids.append(n_idx)
			else:
				neg_ids.append(n_idx)
		
		# Tạo bản ghi cho mỗi lượt click
		for p_id in pos_ids:
			converted_data.append({
				'user_idx': u_idx,
				'history': hist,
				'pos_id': p_id,
				'neg_candidates': np.array(neg_ids, dtype=np.int32)
			})
	
	with open(save_path, 'wb') as f:
		pickle.dump(converted_data, f)

def collate_fn(batch):
	user_indices = torch.stack([item['user_idx'] for item in batch])
	histories = torch.stack([item['history'] for item in batch])
	candidates = torch.stack([item['candidates'] for item in batch])
	labels = torch.stack([item['label'] for item in batch])

	all_news = torch.cat([histories.view(-1), candidates.view(-1)])
	seed_nodes = torch.unique(all_news)
	seed_nodes = seed_nodes[seed_nodes != 0]

	return {
		'user_idx': user_indices,
		'history': histories,
		'candidates': candidates,
		'label': labels,
		'seed_nodes': seed_nodes
	}

@torch.no_grad()
def precompute_bert_features(bert_model, news_tokens, device):
	bert_model.to(device).eval()
	all_bert_outputs = []
	
	for i in tqdm(range(0, len(news_tokens), 64)):
		batch = torch.tensor(news_tokens[i:i+64]).to(device)
		mask = (batch != 0).long()

		out = bert_model(batch, attention_mask=mask).last_hidden_state 
		all_bert_outputs.append(out.cpu()) 
		
	return torch.cat(all_bert_outputs, dim=0) 