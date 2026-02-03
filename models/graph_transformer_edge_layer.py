import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_sum
import numpy as np

class MultiHeadAttentionLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__(aggr='add', node_dim=0)
        self.out_dim = out_dim
        self.num_heads = num_heads
        # Linear 변환
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        
    def forward(self, x, edge_index, edge_attr):
        Q_h = self.Q(x).view(-1, self.num_heads, self.out_dim)
        K_h = self.K(x).view(-1, self.num_heads, self.out_dim)
        V_h = self.V(x).view(-1, self.num_heads, self.out_dim)
        proj_e = self.proj_e(edge_attr).view(-1, self.num_heads, self.out_dim)
        
        # attention scores 계산 및 저장
        attention_scores = self.get_attention_scores(Q_h, K_h, proj_e, edge_index, x.size(0))
        
        out = self.propagate(edge_index, 
                           x=(x, x),
                           Q=Q_h,
                           K=K_h,
                           V=V_h,
                           E=proj_e,
                           attention_scores=attention_scores)
                           
        return out, edge_attr, attention_scores
        
    def message(self, Q_i, K_j, V_j, E, attention_scores, index, ptr, size_i):
        score = attention_scores.view(-1, self.num_heads, 1)
        return V_j * score
        
    def get_attention_scores(self, Q, K, E, edge_index, num_nodes):
        # 1. 기본 Score 계산 (Dot product)
        score = (Q[edge_index[0]] * K[edge_index[1]]).sum(dim=-1) / np.sqrt(self.out_dim)

        # 2. Edge Feature 반영 (Element-wise multiplication)
        score = score.unsqueeze(-1)  # [num_edges, num_heads, 1]
        score = score * E  # [num_edges, num_heads, out_dim]

        # 3. 소프트맥스를 위한 Score 합산
        score = score.sum(dim=-1)  # [num_edges, num_heads]

        # 4. Score 안정화 (Clamping)
        score = torch.clamp(score, -10, 10)
        score_exp = torch.exp(score)

        # 5. Softmax 분모 계산 (scatter_sum 활용)
        src_index, dst_index = edge_index
        denominator = scatter_sum(score_exp, src_index, dim=0, dim_size=num_nodes)  # 노드 개수에 맞게 정규화
        attention = score_exp / (denominator[src_index] + 1e-8)  # 0으로 나누는 것 방지

        return attention


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout, layer_norm=False, 
                 batch_norm=True, residual=True, use_bias=False):
        super().__init__()
        
        self.in_channels = in_dim # input feature 李⑥썝 
        self.out_channels = out_dim # output feature 李⑥썝
        self.num_heads = num_heads # attention head ?닔
        self.dropout = dropout # drop out 鍮꾩쑉
        self.residual = residual #residual connectiuon ?궗?슜 ?뿬遺?
        self.layer_norm = layer_norm #Layer Normalization ?궗?슜 ?뿬遺?
        self.batch_norm = batch_norm 
        
        # attention layer
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)
        
        if layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)
            
        if batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)
        
        # Feed Forward Network Layer
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)
        
    def forward(self, x, edge_index, edge_attr):
        h_in1, e_in1 = x, edge_attr 
        
        # Multi-head attention with attention scores
        h_attn_out, e_attn_out, attention_scores = self.attention(x, edge_index, edge_attr)
        
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        h = self.O_h(h)
        e = self.O_e(e)
        
        if self.residual:
            h = h_in1 + h
            e = e_in1 + e
            
        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)
        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)
            
        h_in2, e_in2 = h, e
        
        h = self.FFN_h_layer1(h)
        h = F.elu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)
        
        e = self.FFN_e_layer1(e)
        e = F.elu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)
        
        if self.residual:
            h = h_in2 + h
            e = e_in2 + e
            
        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)
        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)
            
        return h, e, attention_scores