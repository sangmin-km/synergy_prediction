# SynG2Net: Synergistic Drug Combination Prediction

## Description
A graph transformer-based deep learning framework for predicting synergistic anticancer drug combinations. The model integrates molecular graph representations with gene expression profiles to predict drug synergy scores and identify key genes contributing to synergistic effects.

## Requirements

### Environment Setup
```bash
conda env create -f environment.yaml
conda activate syng2net
```

### Dependencies
- Python: 3.9.19
- PyTorch: 2.0.1+cu118
- PyTorch Geometric: 2.3.1
- torch-scatter: 2.1.2+pt20cu118
- RDKit: 2024.09.2
- NumPy: 1.24.3
- Pandas: 2.0.3
- Scikit-learn: 1.3.0
- SciPy: 1.10.1
- tqdm: 4.67.0

## Usage

### 1. Dataset Creation
```bash
python create_dataset.py
```

### 2. Model Training
```bash
python model_evaluation.py
```

### 3. Gene Importance Analysis
```bash
# Example: Analyze drug pair using PubChem CIDs
# --cid1: PubChem CID for Drug A
# --cid2: PubChem CID for Drug B
python gene_analysis.py --cid1 3385 --cid2 11960529
```

### 4. Attention Visualization
```bash
# Example: Visualize attention scores using PubChem CIDs
python attention_analysis.py --cid1 3385 --cid2 11960529
```

## Contact
- Sangmin Kim: ksm980226@naver.com
- Sunyong Yoo: syyoo@jnu.ac.kr
