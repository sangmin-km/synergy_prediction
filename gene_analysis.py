# -*- coding: utf-8 -*-
"""
Drug Synergy Gene Importance Analysis
"""

import torch
import numpy as np
import pandas as pd
import os
import argparse
from models.net import DrugDrugInteractionNet


class DrugDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_list):
        super(DrugDataset, self).__init__()
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


def find_sample_index(extract_df, drug_a_id, drug_b_id):
    matches = extract_df[
        (extract_df['drug_row_cid'] == drug_a_id) & 
        (extract_df['drug_col_cid'] == drug_b_id)
    ]
    
    if len(matches) == 0:
        return None
    
    return matches.index[0]


def load_gene_names(cell_line='VCAP', data_path='data/data/gene_expression'):
    cell_file = f"{data_path}/{cell_line}.csv"
    
    if not os.path.exists(cell_file):
        return [f"GENE_{i+1:04d}" for i in range(978)]
    
    cell_data = pd.read_csv(cell_file)
    gene_names = cell_data.iloc[:, 0].tolist()
    return gene_names


def extract_top_genes(weights, gene_names, top_n=100):
    if isinstance(weights, torch.Tensor):
        weights_np = weights.cpu().numpy()
    else:
        weights_np = weights
    
    if len(weights_np.shape) > 1:
        weights_np = weights_np[0]
    
    if weights_np.shape[0] != len(gene_names):
        min_size = min(weights_np.shape[0], len(gene_names))
        weights_np = weights_np[:min_size]
        gene_names = gene_names[:min_size]
    
    sorted_indices = np.argsort(weights_np)[-top_n:][::-1]
    
    top_genes = []
    for idx in sorted_indices:
        gene_name = gene_names[idx]
        weight = float(weights_np[idx])
        top_genes.append((gene_name, weight))
    
    return top_genes


def analyze_gene_importance(output, glu1_weights, glu2_weights, gene_names, top_n=10):
    predicted_score = float(output.item())
    
    top_genes_glu1 = extract_top_genes(glu1_weights, gene_names, top_n=100)
    top_genes_glu2 = extract_top_genes(glu2_weights, gene_names, top_n=100)
    
    all_genes = []
    seen_genes = set()
    
    for gene, weight in top_genes_glu1:
        if gene not in seen_genes:
            all_genes.append((gene, weight))
            seen_genes.add(gene)
    
    for gene, weight in top_genes_glu2:
        if gene not in seen_genes:
            all_genes.append((gene, weight))
            seen_genes.add(gene)
    
    all_genes.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'predicted_score': predicted_score,
        'top_genes_glu1': top_genes_glu1,
        'top_genes_glu2': top_genes_glu2,
        'top_important_genes': all_genes[:top_n]
    }


def print_gene_analysis(result, drug_a_name="Drug A", drug_b_name="Drug B"):
    print(f"\n1. Synergy Score: {result['predicted_score']:.4f}")
    
    print(f"\n2. Top Important Genes:")
    for i, (gene, weight) in enumerate(result['top_important_genes'], 1):
        print(f"   {i}. {gene} ({weight:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Drug Synergy Gene Importance Analysis')
    parser.add_argument('--cid1', type=int, required=True, help='Drug A compound ID')
    parser.add_argument('--cid2', type=int, required=True, help='Drug B compound ID')
    parser.add_argument('--model', type=str, default='best_model_fold_1.pt', help='Model path')
    parser.add_argument('--cell_line', type=str, default='VCAP', help='Cell line name')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading data...")
    extract_df = pd.read_csv('data/data/drugdrug_extract.csv')
    
    idx = find_sample_index(extract_df, args.cid1, args.cid2)
    
    if idx is None:
        print(f"Error: Drug pair (CID1={args.cid1}, CID2={args.cid2}) not found in dataset")
        return
    
    print(f"Found sample at index: {idx}")
    
    model = DrugDrugInteractionNet(
        hidden_dim=128,
        gt_layers=3,
        gt_heads=4,
        dropout=0.4
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    model.set_analysis_mode(True)
    
    gene_names = load_gene_names(cell_line=args.cell_line)
    print(f"Loaded {len(gene_names)} gene names")
    
    drug_a_dataset = torch.load('drug_datasets/drug_a_dataset.pt')
    drug_b_dataset = torch.load('drug_datasets/drug_b_dataset.pt')
    
    data_a, data_b = drug_a_dataset[idx], drug_b_dataset[idx]
    
    if not hasattr(data_a, 'batch') or data_a.batch is None:
        data_a.batch = torch.zeros(data_a.num_nodes, dtype=torch.long)
    if not hasattr(data_b, 'batch') or data_b.batch is None:
        data_b.batch = torch.zeros(data_b.num_nodes, dtype=torch.long)
    
    data_a = data_a.to(device)
    data_b = data_b.to(device)
    
    with torch.no_grad():
        output, analysis_data = model(data_a, data_b)
    
    drug_a_id = extract_df.iloc[idx]['drug_row_cid']
    drug_b_id = extract_df.iloc[idx]['drug_col_cid']
    drug_a_name = extract_df.iloc[idx]['drug_row']
    drug_b_name = extract_df.iloc[idx]['drug_col']
    
    print(f"\nDrug Pair: {drug_a_name} (CID: {drug_a_id}) + {drug_b_name} (CID: {drug_b_id})")
    
    result = analyze_gene_importance(
        output=output,
        glu1_weights=analysis_data['glu1_weights'],
        glu2_weights=analysis_data['glu2_weights'],
        gene_names=gene_names,
        top_n=10
    )
    
    print_gene_analysis(result, drug_a_name=drug_a_name, drug_b_name=drug_b_name)


if __name__ == "__main__":
    main()