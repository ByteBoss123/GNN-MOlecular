# Graph Neural Networks vs Fingerprints for Molecular Property Prediction

Do graph neural networks beat a random forest on molecular fingerprints? Tested on two MoleculeNet benchmarks with
a **scaffold split**: test molecules have ring skeletons never seen in training, the setting that resembles
screening new chemistry and the one where models usually fall short.

**Data** (`data/download.sh`, SHA-256 checked): ESOL, 1,128 molecules with measured water solubility (log mol/L);
Tox21, 8,006 molecules with 12 toxicity assays (83% of molecule–assay labels present).

**Models.** GCN (mean pooling) and GIN (sum pooling, more expressive) on atom graphs with 37 atom features;
random forest and XGBoost on 2,048-bit Morgan fingerprints (radius 2). Each run uses 3 seeds; GNNs train for 100
epochs with the epoch picked on validation, and the test set is scored once.

## Results (`results/benchmark.json`, mean ± std over 3 seeds, CPU)

| Model | ESOL RMSE ↓ | ESOL R² | Tox21 mean ROC-AUC ↑ |
|---|---|---|---|
| **GCN** | **1.014 ± 0.004** | **0.770** | **0.755 ± 0.002** |
| GIN | 1.031 ± 0.087 | 0.761 | 0.738 ± 0.005 |
| Random forest (fingerprints) | 1.476 ± 0.008 | 0.513 | 0.752 ± 0.001 |
| XGBoost (fingerprints) | 1.367 ± 0.028 | 0.582 | 0.748 ± 0.000 |

- **Solubility: the GNN clearly wins**, cutting RMSE 26% versus the best fingerprint model (1.01 vs 1.37).
  Solubility depends on whole-molecule properties such as size and polarity, which message passing aggregates and
  a bag of substructure bits does not.
- **Toxicity: no real gain.** GCN's 0.755 is within noise of the random forest's 0.752, which trains in about half
  the time. Toxicity often hinges on specific substructures, exactly what fingerprints encode.
- GIN, the more expressive model, did not beat GCN on either task at this data size.

## What changed in the October 2026 rebuild

The previous README reported toxicity AUC 0.87, solubility R² 0.82 and "+15% accuracy over baselines" for "8,000+
compounds". The repo held only `main.py` and `config.py`, which imported modules that did not exist, so those
numbers could not have come from this code. They were replaced by the benchmark above, run from scratch with a
harder split than the random split those figures implied.

## Run

```
pip install -r requirements.txt
bash data/download.sh
python src/train.py          # ~25 min on 2 CPU cores; EPOCHS=10 for a quick check
```

`src/data.py` holds featurization and the scaffold split; `src/train.py` holds the models and the benchmark.
