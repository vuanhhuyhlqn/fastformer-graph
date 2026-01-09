import torch.nn as nn

class Fastformer_graph(nn.Module):
    def __init__(self, news_encoder, user_encoder, gnn):
        super().__init__()
        self.news_encoder = news_encoder
        self.user_enoder = user_encoder
        self.gnn = gnn
    
    def forward(self, batch):
        # Content embedding
        content_emb = self.news_encoder(batch.x)

        # Structure embedding
        struct_emb = self.gnn(content_emb, batch.edge_index)

        news_emb = content_emb + struct_emb

        

        return
