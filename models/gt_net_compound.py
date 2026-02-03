import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from .graph_transformer_edge_layer import GraphTransformerLayer

class GraphTransformer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim, n_layers, n_heads, dropout):
        super(GraphTransformer, self).__init__()
        
        self.layer_norm = False
        self.batch_norm = True
        self.residual = True
        
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.pos_encoder = nn.Linear(8, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim, hidden_dim, n_heads, dropout,
                self.layer_norm, self.batch_norm, self.residual
            ) for _ in range(n_layers-1)
        ])
        
        self.layers.append(
            GraphTransformerLayer(
                hidden_dim, out_dim, n_heads, dropout,
                self.layer_norm, self.batch_norm, self.residual
            )
        )
        
    def forward(self, data):
        h = data.x
        e = data.edge_attr
        edge_index = data.edge_index
        pos_enc = data.pos_enc
        
        h = self.node_encoder(h)
        e = self.edge_encoder(e)
        pos_enc = self.pos_encoder(pos_enc)
        
        h = h + pos_enc
        h = self.dropout(h)
        
        attention_scores_all = []
        
        for layer in self.layers:
            h, e, attention_scores = layer(h, edge_index, e)
            attention_scores_all.append(attention_scores)
        
        return h, e, attention_scores_all