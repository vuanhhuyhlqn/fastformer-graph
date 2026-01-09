import torch.nn as nn
from torch_geometric.loader import NeighborLoader
class Fastformer_graph(nn.Module):
    def __init__(self, news_encoder, user_encoder, gnn):
        super().__init__()
        self.news_encoder = news_encoder
        self.user_enoder = user_encoder
        self.gnn = gnn
    
    def forward(self, batch):
        return
