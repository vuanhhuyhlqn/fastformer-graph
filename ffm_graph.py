import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from transformers import BertModel
from fastformer import FastformerEncoder

class NewsEncoder(nn.Module):
    def __init__(self, fastformer_config, embedding_dim=256):
        super(NewsEncoder, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.project = nn.Linear(768, embedding_dim)
        self.fastformer = FastformerEncoder(fastformer_config)

    def forward(self, x):
        with torch.no_grad():
            bert_output = self.bert(x).last_hidden_state
        word_vecs = self.project(bert_output)
        news_vec = self.fastformer(word_vecs)
        return news_vec

class Fastformer_graph(nn.Module):
    def __init__(self, news_encoder, user_encoder, gnn):
        super().__init__()
        self.news_encoder = news_encoder
        self.user_enoder = user_encoder
        self.gnn = gnn
    
    def forward(self, batch, sub_graph, device):
        news_feature = self.news_encoder(sub_graph.x)

