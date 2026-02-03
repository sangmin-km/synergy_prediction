# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from . import gt_net_compound

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DrugDrugInteractionNet(nn.Module):
    def __init__(self, hidden_dim=128, gt_layers=3, gt_heads=4, dropout=0.3):
        super(DrugDrugInteractionNet, self).__init__()
        
        self.graph_transformer = gt_net_compound.GraphTransformer(
            node_dim=24,
            edge_dim=10,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=gt_layers,
            n_heads=gt_heads,
            dropout=dropout
        )
        
        self.gene_shared = nn.Sequential(
            nn.Linear(978, 978),
            nn.ELU(),
            nn.BatchNorm1d(978)
        )
        
        self.glu1 = nn.Linear(978 * 2, 978)
        self.glu2 = nn.Linear(978 * 2, 978)
        
        self.bn_gene1 = nn.BatchNorm1d(978)
        self.bn_gene2 = nn.BatchNorm1d(978)
        
        self.gene_shared2 = nn.Sequential(
            nn.Linear(978, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self.analysis_mode = False
        
    def process_gene_expression(self, gene_expr_a, gene_expr_b):
        gene_expr_a = gene_expr_a.view(-1, 978)
        gene_expr_b = gene_expr_b.view(-1, 978)
        
        drug1 = self.gene_shared(gene_expr_a)
        drug2 = self.gene_shared(gene_expr_b)
        
        concat = torch.cat([drug1, drug2], dim=1)
        
        glu1 = torch.sigmoid(self.glu1(concat))
        glu2 = torch.sigmoid(self.glu2(concat))
        
        # Store GLU weights for analysis
        self.glu1_weights = glu1
        self.glu2_weights = glu2
        
        drug1_selected = drug1 * glu1
        drug2_selected = drug2 * glu2
        
        drug1_selected = self.bn_gene1(drug1_selected)
        drug2_selected = self.bn_gene2(drug2_selected)
        
        drug1_emb = self.gene_shared2(drug1_selected)
        drug2_emb = self.gene_shared2(drug2_selected)
        
        return drug1_emb, drug2_emb
    
    def set_analysis_mode(self, mode=True):
        self.analysis_mode = mode
        return self
    
    def forward(self, data_a, data_b):
        # GraphTransformer returns (node_features, edge_features, attention_scores_all)
        node_feat_a, edge_feat_a, attn_scores_a = self.graph_transformer(data_a)
        node_feat_b, edge_feat_b, attn_scores_b = self.graph_transformer(data_b)
        
        # Apply global_mean_pool to node features
        graph_feat_a = global_mean_pool(node_feat_a, data_a.batch)
        graph_feat_b = global_mean_pool(node_feat_b, data_b.batch)

        gene_feat_a, gene_feat_b = self.process_gene_expression(
            data_a.express, data_b.express
        )
        
        combined_feats = torch.cat([
            graph_feat_a, graph_feat_b,
            gene_feat_a, gene_feat_b
        ], dim=1)
        
        output = self.classifier(combined_feats)
        
        # Return attention scores only in analysis mode
        if self.analysis_mode:
            return output, {
                'attn_scores_a': attn_scores_a,
                'attn_scores_b': attn_scores_b,
                'node_feat_a': node_feat_a,
                'node_feat_b': node_feat_b,
                'edge_feat_a': edge_feat_a,
                'edge_feat_b': edge_feat_b,
                'glu1_weights': self.glu1_weights,
                'glu2_weights': self.glu2_weights
            }
        
        # Return output only in training mode
        return output