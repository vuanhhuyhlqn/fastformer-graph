import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from transformers import BertModel
from fastformer import FastformerEncoder

class NewsEncoder(nn.Module):
    def __init__(self, fastformer_config, bert_features_path):
        super(NewsEncoder, self).__init__()
        self.bert_features = torch.load(bert_features_path)
        self.project = nn.Linear(312, fastformer_config.hidden_size)
        self.fastformer = FastformerEncoder(fastformer_config)

    def forward(self, news_ids, token_ids, device):
        news_ids_cpu = news_ids.cpu()

        x = self.bert_features[news_ids_cpu].to(device)
        mask = (token_ids != 0).long().to(device)

        word_vecs = self.project(x) # [num_nodes, 128, 256]
        news_vec = self.fastformer(word_vecs, attention_mask=mask) # [num_nodes, 256]

        return news_vec

class UserEncoder(nn.Module):
    def __init__(self, fastformer_config):
        super(UserEncoder, self).__init__()
        self.fastformer = FastformerEncoder(fastformer_config)

    def forward(self, history_embs, history_mask):
        # history_embs: [batch_size, 50, 256]
        # history_mask: [batch, 50]
        user_vec = self.fastformer(history_embs, history_mask)
        return user_vec

class Fastformer_Graph(nn.Module):
    def __init__(self, news_encoder, user_encoder, gnn):
        super().__init__()
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
        self.gnn = gnn
    
    def forward(self, batch, sub_graph, device):
        news_emb = self.news_encoder(sub_graph.n_id, sub_graph.x, device) # [num_nodes, 128] -> [num_nodes, 256]
        struct_emb = self.gnn(news_emb, sub_graph.edge_index)

        x = news_emb + struct_emb

        max_id = sub_graph.n_id.max() + 1
        news_lookup = torch.zeros(max_id, x.size(1)).to(device)
        news_lookup[sub_graph.n_id] = x

        history_embs = news_lookup[batch['history']] # [batch, 32, 256]
        history_mask = (batch['history'] != 0) 
        user_vec = self.user_encoder(history_embs, history_mask)

        cand_embs = news_lookup[batch['candidates']]

        scores = torch.bmm(cand_embs, user_vec.unsqueeze(-1)).squeeze(-1)

        return scores







