import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from transformers import BertModel
from fastformer import FastformerEncoder

class NewsEncoder(nn.Module):
    def __init__(self, fastformer_config):
        super(NewsEncoder, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.project = nn.Linear(768, fastformer_config.hidden_size)
        self.fastformer = FastformerEncoder(fastformer_config)

    def forward(self, x):
        # x : [num_nodes, 128]
        mask = (x != 0).long() # [num_nodes, 128]
        with torch.no_grad():
            bert_output = self.bert(x, attention_mask=mask).last_hidden_state # [num_nodes, 128, 768]

        word_vecs = self.project(bert_output) # [num_nodes, 128, 256]
        news_vec = self.fastformer(word_vecs, attention_mask=mask) # [num_nodes, 256]

        return news_vec

class UserEncoder(nn.Module):
    def __init__(self, fastformer_config):
        super(UserEncoder, self).__init__()
        self.fastformer = FastformerEncoder(fastformer_config)

    def forward(self, history_embs, history_mask):
        # history_embs: [batch_size, 50, 256]
        # history_mask: [batch, 50]
        user_vec = self.fastformer_user(history_embs, history_mask)
        return user_vec

class Fastformer_Graph(nn.Module):
    def __init__(self, news_encoder, user_encoder, gnn):
        super().__init__()
        self.news_encoder = news_encoder
        self.user_enoder = user_encoder
        self.gnn = gnn
    
    def forward(self, batch, sub_graph, device):
        news_emb = self.news_encoder(sub_graph.x) # [num_nodes, 128] -> [num_nodes, 256]
        struct_emb = self.gnn(news_emb, sub_graph.edge_index)

        x = news_emb + struct_emb

        max_id = sub_graph.n_id.max() + 1
        news_lookup = torch.zeros(max_id, x.size(1)).to(device)
        news_lookup[sub_graph.n_id] = x

        history_embs = news_lookup[batch['history']] # [batch, 50, 256]
        history_mask = (batch['history'] != 0) 
        user_vec = self.user_encoder(history_embs, history_mask)

        cand_embs = news_lookup[batch['candidates']]

        scores = torch.bmm(cand_embs, user_vec.unsqueeze(-1)).squeeze(-1)

        return scores







