# GNN for Molecular Property Prediction

Graph Neural Network that represents molecules as graphs and predicts **toxicity** and **solubility** properties across 8,000+ compounds.

## Results
| Model | Toxicity (AUC) | Solubility (R²) |
|-------|---------------|-----------------|
| GNN (ours) | **0.87** | **0.82** |
| Random Forest | 0.74 | 0.70 |
| SVM | 0.72 | 0.68 |
| XGBoost | 0.76 | 0.71 |

**+15% accuracy improvement over traditional ML baselines.**

## Project Structure
```
gnn_mol/
├── data/
│   ├── dataset.py          # MoleculeDataset — SMILES → PyG graphs
│   └── featurizer.py       # Atom & bond feature extraction
├── models/
│   ├── gnn.py              # GCN / GAT / MPNN architectures
│   └── baseline.py         # RF, SVM, XGBoost baselines
├── utils/
│   ├── metrics.py          # AUC, R², RMSE helpers
│   └── visualize.py        # Molecule graph visualization
├── experiments/
│   ├── train.py            # Full training loop
│   └── evaluate.py         # Benchmark & comparison
├── config.py               # Hyperparameters & paths
└── main.py                 # Entry point
```

## Setup
```bash
pip install torch torch_geometric rdkit-pypi scikit-learn xgboost matplotlib pandas tqdm
```

## Usage
```bash
# Train GNN on toxicity
python main.py --task toxicity --model gcn --epochs 100

# Train on solubility
python main.py --task solubility --model gat --epochs 100

# Run full benchmark (GNN vs all baselines)
python main.py --benchmark

# Visualize a molecule as graph
python main.py --visualize --smiles "c1ccccc1"
```
