"""
Drug Synergy Dataset Creation
Converts drug SMILES and gene expression data into graph datasets for training.
"""

import os
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data


class DrugDataset(torch.utils.data.Dataset):
    """Custom dataset class for drug pair data."""
    
    def __init__(self, data_list):
        super(DrugDataset, self).__init__()
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


def smiles_features(mol):
    """
    Extract molecular features from RDKit molecule object.
    Returns: node features (24-dim), edge indices, edge attributes (10-dim)
    """
    symbols = ['B', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S']
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3
    ]
    stereos = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
    
    # Node features
    xs = []
    for atom in mol.GetAtoms():
        symbol = [0.] * len(symbols)
        symbol[symbols.index(atom.GetSymbol())] = 1.
        
        degree = [0.] * 5
        degree[min(atom.GetDegree(), 4)] = 1.
        
        hybridization = [0.] * len(hybridizations)
        hyb_type = atom.GetHybridization()
        if hyb_type in hybridizations:
            hyb_idx = hybridizations.index(hyb_type)
            hybridization[hyb_idx] = 1.
        
        features = (
            symbol +
            degree +
            [atom.GetFormalCharge()] +
            [atom.GetIsAromatic()] +
            hybridization +
            [float(i == atom.GetTotalNumHs()) for i in range(4)] +
            [atom.HasProp('_ChiralityPossible')]
        )
        xs.append(features)

    # Edge features
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
        edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]
        
        edge_attr = [
            bond.GetBondType() == Chem.rdchem.BondType.SINGLE,
            bond.GetBondType() == Chem.rdchem.BondType.DOUBLE,
            bond.GetBondType() == Chem.rdchem.BondType.TRIPLE,
            bond.GetBondType() == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ] + [float(bond.GetStereo() == s) for s in stereos]
        
        edge_attrs += [edge_attr, edge_attr]

    x = torch.tensor(xs, dtype=torch.float)
    
    if edge_indices:
        edge_index = torch.tensor(edge_indices).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 10), dtype=torch.float)
    
    return x, edge_index, edge_attr


def compute_laplacian_pe(adj_matrix, k=8):
    """
    Compute Laplacian positional encoding for graph nodes.
    Args:
        adj_matrix: Adjacency matrix (numpy array)
        k: Number of eigenvectors to use
    Returns:
        Positional encoding tensor (N x k)
    """
    N = adj_matrix.shape[0]
    D = np.diag(adj_matrix.sum(axis=1))
    L = D - adj_matrix
    
    D_inv_sqrt = np.linalg.inv(np.sqrt(D + np.eye(N) * 1e-8))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    try:
        eigval, eigvec = np.linalg.eigh(L_norm)
        idx = eigval.argsort()[:k]
        return torch.from_numpy(eigvec[:, idx]).float()
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Laplacian eigenvalue decomposition failed: {str(e)}")


def smiles_to_graph(smiles):
    """Convert SMILES string to graph representation."""
    mol = Chem.MolFromSmiles(smiles)   
    if mol is None:
        return None
    return smiles_features(mol)


def load_integrated_data(extract_file, smiles_file, cell_line_files, cell_line_path, score="S"):
    """
    Load and integrate drug synergy data with molecular graphs and gene expression.
    
    Args:
        extract_file: Path to drug combination data CSV
        smiles_file: Path to drug SMILES data CSV
        cell_line_files: List of cell line names
        cell_line_path: Path to gene expression data directory
        score: Score column name (default: "S")
    
    Returns:
        Tuple of (drug_a_dataset, drug_b_dataset)
    """
    # Load data
    extract = pd.read_csv(extract_file, usecols=[3,4,5,6,7,8,9,10])
    smiles_data = pd.read_csv(smiles_file)
    
    all_express = {}
    for cell in cell_line_files:
        cell_data = pd.read_csv(f"{cell_line_path}{cell}.csv")
        all_express[cell] = cell_data
    
    drug_a_data = []
    drug_b_data = []
    
    # Process each drug combination
    for idx, row in extract.iterrows():
        try:
            drug_a_id = row['drug_row_cid']
            drug_b_id = row['drug_col_cid']
            
            # Get SMILES
            smiles_a = smiles_data.loc[smiles_data['drug_id'] == drug_a_id, 'smiles'].values
            smiles_b = smiles_data.loc[smiles_data['drug_id'] == drug_b_id, 'smiles'].values
            
            if len(smiles_a) == 0 or len(smiles_b) == 0:
                continue
            
            # Convert to graphs
            graph_a = smiles_to_graph(smiles_a[0])
            graph_b = smiles_to_graph(smiles_b[0])
            
            if graph_a is None or graph_b is None:
                continue
            
            # Get gene expression data
            cell_line = row['cell_line_name']
            if cell_line not in all_express:
                continue
                
            expr_data = all_express[cell_line]
            
            if str(drug_a_id) not in expr_data.columns or str(drug_b_id) not in expr_data.columns:
                continue
            
            expr_a = torch.tensor(expr_data[str(drug_a_id)].values, dtype=torch.float32)
            expr_b = torch.tensor(expr_data[str(drug_b_id)].values, dtype=torch.float32)
            label = torch.tensor([row[score]], dtype=torch.float32)
            
            if torch.isnan(expr_a).any() or torch.isnan(expr_b).any() or torch.isnan(label).any():
                continue
            
            # Compute adjacency matrices and positional encodings
            adj_a = torch.zeros((graph_a[0].size(0), graph_a[0].size(0)))
            adj_a[graph_a[1][0], graph_a[1][1]] = 1
            pos_enc_a = compute_laplacian_pe(adj_a.numpy())
            
            adj_b = torch.zeros((graph_b[0].size(0), graph_b[0].size(0)))
            adj_b[graph_b[1][0], graph_b[1][1]] = 1
            pos_enc_b = compute_laplacian_pe(adj_b.numpy())
            
            # Create PyG Data objects
            data_a = Data(
                x=graph_a[0],
                edge_index=graph_a[1],
                edge_attr=graph_a[2],
                pos_enc=pos_enc_a,
                express=expr_a,
                y=label,
                num_nodes=graph_a[0].size(0)
            )
            
            data_b = Data(
                x=graph_b[0],
                edge_index=graph_b[1],
                edge_attr=graph_b[2],
                pos_enc=pos_enc_b,
                express=expr_b,
                y=label,
                num_nodes=graph_b[0].size(0)
            )
            
            drug_a_data.append(data_a)
            drug_b_data.append(data_b)
            
        except Exception as e:
            continue
    
    return DrugDataset(drug_a_data), DrugDataset(drug_b_data)


if __name__ == '__main__':
    # Data paths
    paths = {
        'extract': "data/data/drugdrug_extract.csv",
        'cell_line': "data/data/gene_expression/",
        'smiles': "data/data/smiles.csv"
    }
    
    cell_lines = ["A375", "A549", "HCT116", "HS 578T", "HT29", "LNCAP", "LOVO",
                 "MCF7", "PC-3", "RKO", "SK-MEL-28", "SW-620", "VCAP"]
    
    # Create datasets
    print("Creating datasets...")
    drug_a_dataset, drug_b_dataset = load_integrated_data(
        paths['extract'], 
        paths['smiles'], 
        cell_lines, 
        paths['cell_line']
    )
    
    # Save datasets
    output_dir = "drug_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(drug_a_dataset, f"{output_dir}/drug_a_dataset.pt")
    torch.save(drug_b_dataset, f"{output_dir}/drug_b_dataset.pt")
    
    print(f"Dataset created: {len(drug_a_dataset)} samples")
    print(f"Saved to {output_dir}/")