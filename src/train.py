"""Benchmark: GCN and GIN vs random forest and XGBoost on Morgan fingerprints, ESOL and Tox21, scaffold split.

Each neural model is trained with 3 seeds; the epoch is chosen on the validation set and the test set is scored
once. Writes results/benchmark.json and results/benchmark.csv.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, global_mean_pool
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import N_ATOM_FEATURES, TOX21_TASKS, fingerprint, load, mol_to_graph, scaffold_split  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2]
EPOCHS = int(os.environ.get("EPOCHS", 100))
torch.set_num_threads(max(1, os.cpu_count() or 1))


class GNN(nn.Module):
    def __init__(self, arch, n_out, hidden=128, layers=3, dropout=0.2):
        super().__init__()
        self.arch, self.convs, self.norms = arch, nn.ModuleList(), nn.ModuleList()
        for i in range(layers):
            d_in = N_ATOM_FEATURES if i == 0 else hidden
            if arch == "gcn":
                self.convs.append(GCNConv(d_in, hidden))
            else:  # GIN: sum aggregation with an MLP, more expressive than GCN's mean
                self.convs.append(GINConv(nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, b):
        h = b.x
        for conv, norm in zip(self.convs, self.norms):
            h = self.drop(torch.relu(norm(conv(h, b.edge_index))))
        pool = global_mean_pool if self.arch == "gcn" else global_add_pool
        return self.head(pool(h, b.batch))


def score(task, y_true, y_pred):
    if task == "esol":
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return {"rmse": round(rmse, 4), "r2": round(float(r2_score(y_true, y_pred)), 4)}
    aucs = []
    for t in range(y_true.shape[1]):
        m = ~np.isnan(y_true[:, t])
        if m.sum() and len(np.unique(y_true[m, t])) == 2:
            aucs.append(roc_auc_score(y_true[m, t], y_pred[m, t]))
    return {"mean_roc_auc": round(float(np.mean(aucs)), 4), "tasks_scored": len(aucs)}


def primary(task, s):
    return -s["rmse"] if task == "esol" else s["mean_roc_auc"]


def predict(model, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for b in loader:
            out.append(model(b))
    return torch.cat(out).numpy()


def run_gnn(task, arch, graphs, y, split, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va, te = split
    n_out = 1 if task == "esol" else len(TOX21_TASKS)
    mu, sd = (y[tr].mean(), y[tr].std()) if task == "esol" else (0.0, 1.0)  # standardize regression target
    for g, v in zip(graphs, y):
        g.y = torch.tensor([(v - mu) / sd] if task == "esol" else [v], dtype=torch.float).view(1, -1)
    model = GNN(arch, n_out)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    g_tr = torch.Generator().manual_seed(seed)
    train_loader = DataLoader([graphs[i] for i in tr], batch_size=64, shuffle=True, generator=g_tr)
    val_loader = DataLoader([graphs[i] for i in va], batch_size=256)
    test_loader = DataLoader([graphs[i] for i in te], batch_size=256)
    best, best_state, best_epoch = -np.inf, None, 0
    for epoch in range(EPOCHS):
        model.train()
        for b in train_loader:
            opt.zero_grad()
            out = model(b)
            if task == "esol":
                loss = nn.functional.mse_loss(out, b.y)
            else:  # masked multi-task BCE: Tox21 labels are missing for many molecule-task pairs
                m = ~torch.isnan(b.y)
                loss = nn.functional.binary_cross_entropy_with_logits(out[m], b.y[m])
            loss.backward()
            opt.step()
        p = predict(model, val_loader)
        p = p.ravel() * sd + mu if task == "esol" else p
        s = primary(task, score(task, y[va], p))
        if s > best:
            best, best_epoch = s, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    p = predict(model, test_loader)
    p = p.ravel() * sd + mu if task == "esol" else p
    return score(task, y[te], p), best_epoch


def run_baseline(task, kind, fps, y, split, seed):
    tr, va, te = split
    trv = np.concatenate([tr, va])  # no tuning, so validation data joins training
    if task == "esol":
        m = RandomForestRegressor(500, random_state=seed, n_jobs=-1) if kind == "rf" else \
            XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=seed)
        m.fit(fps[trv], y[trv])
        return score(task, y[te], m.predict(fps[te]))
    pred = np.full((len(te), len(TOX21_TASKS)), np.nan)
    for t in range(len(TOX21_TASKS)):
        lab = ~np.isnan(y[trv, t])
        m = RandomForestClassifier(500, random_state=seed, n_jobs=-1, class_weight="balanced") if kind == "rf" else \
            XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=seed)
        m.fit(fps[trv][lab], y[trv][lab, t].astype(int))
        pred[:, t] = m.predict_proba(fps[te])[:, 1]
    return score(task, y[te], pred)


def main():
    rows, meta = [], {}
    for task in ("esol", "tox21"):
        mols, y, smiles = load(task)
        split = scaffold_split(smiles)
        meta[task] = {"molecules": len(mols), "train": len(split[0]), "val": len(split[1]), "test": len(split[2])}
        if task == "tox21":
            meta[task]["label_coverage"] = round(float((~np.isnan(y)).mean()), 3)
        graphs = [mol_to_graph(m, 0.0) for m in mols]
        fps = np.array([fingerprint(m) for m in mols])
        for kind in ("rf", "xgb"):
            for seed in SEEDS:
                t0 = time.time()
                rows.append({"task": task, "model": kind, "seed": seed, **run_baseline(task, kind, fps, y, split, seed),
                             "seconds": round(time.time() - t0, 1)})
                print(rows[-1], flush=True)
        for arch in ("gcn", "gin"):
            for seed in SEEDS:
                t0 = time.time()
                s, ep = run_gnn(task, arch, graphs, y, split, seed)
                rows.append({"task": task, "model": arch, "seed": seed, **s, "best_epoch": ep,
                             "seconds": round(time.time() - t0, 1)})
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "results", "benchmark.csv"), index=False)
    summary = {"data": meta, "epochs": EPOCHS, "seeds": SEEDS, "results": {}}
    for task, metric in (("esol", "rmse"), ("esol", "r2"), ("tox21", "mean_roc_auc")):
        g = df[df.task == task].groupby("model")[metric]
        summary["results"][f"{task}_{metric}"] = {m: [round(float(v.mean()), 4), round(float(v.std()), 4)] for m, v in g}
    with open(os.path.join(ROOT, "results", "benchmark.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
