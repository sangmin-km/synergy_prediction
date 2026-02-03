# -*- coding: utf-8 -*-
"""
Drug Synergy Attention Score Analysis and Visualization
"""

import torch
import numpy as np
import pandas as pd
import os
import argparse
from models.net import DrugDrugInteractionNet
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor


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


def calculate_node_importance(edge_index, edge_attn, num_nodes):
    node_importance = torch.zeros(num_nodes)
    node_count = torch.zeros(num_nodes)
    
    for i in range(min(edge_index.shape[1], len(edge_attn))):
        dst = edge_index[1, i].item()
        if dst < num_nodes:
            node_importance[dst] += edge_attn[i].item()
            node_count[dst] += 1
    
    node_count[node_count == 0] = 1
    node_importance = node_importance / node_count
    
    if node_importance.max() > node_importance.min():
        node_importance = (node_importance - node_importance.min()) / \
                         (node_importance.max() - node_importance.min())
    
    return node_importance


def visualize_molecule_attention(mol, node_importance, output_path):
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
    
    atom_colors = {i: (1.0, 1.0 - node_importance[i].item(), 1.0 - node_importance[i].item())
                   for i in range(min(len(node_importance), mol.GetNumAtoms()))}
    
    drawer.DrawMolecule(
        mol, 
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBonds=[],
        highlightAtomRadii={i: 0.3 for i in atom_colors.keys()}
    )
    drawer.FinishDrawing()
    
    with open(output_path, 'wb') as f:
        f.write(drawer.GetDrawingText())


def analyze_attention(smiles, attn_scores, edge_index, edge_mask, node_mask, output_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Failed to create molecule from SMILES: {smiles}")
        return False
    
    num_nodes = node_mask.sum().item()
    sample_edge_index = edge_index[:, edge_mask].cpu()
    
    all_attns = [layer[edge_mask].mean(dim=1).cpu() for layer in attn_scores]
    min_len = min(len(a) for a in all_attns)
    overall = torch.mean(torch.stack([a[:min_len] for a in all_attns]), dim=0)
    
    node_imp = calculate_node_importance(sample_edge_index[:, :min_len], overall, num_nodes)
    visualize_molecule_attention(mol, node_imp, output_path)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Drug Synergy Attention Analysis')
    parser.add_argument('--cid1', type=int, required=True, help='Drug A compound ID')
    parser.add_argument('--cid2', type=int, required=True, help='Drug B compound ID')
    parser.add_argument('--model', type=str, default='best_model_fold_1.pt', help='Model path')
    parser.add_argument('--output', type=str, default='attention_visualization', help='Output directory')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading data...")
    extract_df = pd.read_csv('data/drugdrug_extract.csv')
    
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
    
    drug_a_dataset = torch.load('drug_datasets/drug_a_dataset.pt')
    drug_b_dataset = torch.load('drug_datasets/drug_b_dataset.pt')
    
    smiles_df = pd.read_csv('data/smiles.csv')
    smiles_dict = dict(zip(smiles_df['drug_id'], smiles_df['smiles']))
    
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
    
    synergy_score = output.item()
    
    print(f"\nDrug Pair: {drug_a_name} (CID: {drug_a_id}) + {drug_b_name} (CID: {drug_b_id})")
    print(f"Synergy Score: {synergy_score:.4f}")
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    node_mask_a = (data_a.batch == 0)
    node_mask_b = (data_b.batch == 0)
    edge_mask_a = (data_a.batch[data_a.edge_index[0]] == 0)
    edge_mask_b = (data_b.batch[data_b.edge_index[0]] == 0)
    
    smiles_a = smiles_dict.get(drug_a_id, '')
    smiles_b = smiles_dict.get(drug_b_id, '')
    
    if smiles_a:
        output_path_a = os.path.join(output_dir, f"drug_a_{drug_a_id}_{drug_a_name}.png")
        print(f"\nGenerating attention visualization for {drug_a_name}...")
        if analyze_attention(smiles_a, analysis_data['attn_scores_a'],
                           data_a.edge_index, edge_mask_a, node_mask_a, output_path_a):
            print(f"Saved: {output_path_a}")
    
    if smiles_b:
        output_path_b = os.path.join(output_dir, f"drug_b_{drug_b_id}_{drug_b_name}.png")
        print(f"Generating attention visualization for {drug_b_name}...")
        if analyze_attention(smiles_b, analysis_data['attn_scores_b'],
                           data_b.edge_index, edge_mask_b, node_mask_b, output_path_b):
            print(f"Saved: {output_path_b}")
    
    print(f"\nVisualization complete! Images saved to: {output_dir}/")


if __name__ == "__main__":
    main()