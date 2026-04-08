"""
main.py — Entry point for GNN molecular property prediction.

Usage examples:
    # Train GNN on toxicity (Tox21, NR-AR task)
    python main.py --task toxicity --arch gcn --epochs 100

    # Train on solubility (ESOL)
    python main.py --task solubility --arch gat --epochs 100

    # Use MPNN (edge-feature-aware) architecture
    python main.py --task toxicity --arch mpnn

    # Full benchmark: GNN vs RF / SVM / XGBoost
    python main.py --benchmark --task toxicity

    # Visualize a molecule as a graph
    python main.py --visualize --smiles "c1ccccc1O"

    # Inspect an existing checkpoint
    python main.py --eval --checkpoint outputs/toxicity_gcn_best.pt
"""

import argparse
import logging
import os
import sys

import torch

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="GNN Molecular Property Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--benchmark",  action="store_true",
                      help="Run full GNN-vs-baselines benchmark")
    mode.add_argument("--visualize",  action="store_true",
                      help="Visualize a molecule as a graph")
    mode.add_argument("--eval",       action="store_true",
                      help="Evaluate an existing checkpoint on the test set")

    # Data
    p.add_argument("--task",     choices=["toxicity", "solubility"],
                   default="toxicity")
    p.add_argument("--dataset",  default=None,
                   help="Override dataset name (default: tox21/esol per task)")
    p.add_argument("--tox_col",  default="NR-AR",
                   help="Tox21 target column")
    p.add_argument("--data_dir", default="data/raw",
                   help="Root dir for dataset download & cache")

    # Model
    p.add_argument("--arch",       choices=["gcn", "gat", "mpnn"], default="gcn")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--heads",      type=int, default=4,
                   help="Attention heads (GAT only)")
    p.add_argument("--pooling",
                   choices=["mean", "sum", "max", "attention"], default="mean")

    # Training
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--pos_weight",  type=float, default=3.0,
                   help="BCE pos_weight for class imbalance (toxicity)")
    p.add_argument("--early_stop",  type=int,   default=20)

    # Misc
    p.add_argument("--device",      default="auto",
                   help="cuda | mps | cpu | auto")
    p.add_argument("--output_dir",  default="outputs")
    p.add_argument("--checkpoint",  default=None,
                   help="Path to checkpoint for --eval / --benchmark")
    p.add_argument("--smiles",      default="c1ccccc1",
                   help="SMILES string for --visualize")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = resolve_device(args.device)
    logger.info(f"Device: {device}")

    # Default dataset per task
    if args.dataset is None:
        args.dataset = "tox21" if args.task == "toxicity" else "esol"

    run_name = f"{args.task}_{args.arch}"

    # ---- Visualize ----------------------------------------------------------
    if args.visualize:
        from utils.visualize import draw_molecule_as_graph, draw_molecule_2d
        os.makedirs(args.output_dir, exist_ok=True)
        draw_molecule_as_graph(
            args.smiles,
            save_path=os.path.join(args.output_dir, "molecule_graph.png"),
        )
        draw_molecule_2d(
            args.smiles,
            save_path=os.path.join(args.output_dir, "molecule_2d.svg"),
        )
        return

    # ---- Data loaders -------------------------------------------------------
    from data.dataset import get_loaders
    logger.info(f"Loading {args.dataset} ({args.task}) ...")
    train_loader, val_loader, test_loader = get_loaders(
        root=args.data_dir,
        dataset_name=args.dataset,
        task=args.task,
        tox_target_col=args.tox_col,
        batch_size=args.batch_size,
    )

    # Infer feature dims from data
    sample   = next(iter(train_loader))
    in_dim   = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1]
    logger.info(f"in_dim={in_dim}, edge_dim={edge_dim}, "
                f"train_batches={len(train_loader)}")

    # ---- Build model --------------------------------------------------------
    from models.gnn import GNNModel
    model = GNNModel(
        arch=args.arch,
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        heads=args.heads,
        pooling=args.pooling,
    )
    logger.info(f"Model: {model.arch.upper()} | params={model.num_parameters():,}")

    # ---- Eval only ----------------------------------------------------------
    if args.eval:
        if args.checkpoint is None or not os.path.exists(args.checkpoint):
            logger.error("--eval requires --checkpoint <path>")
            sys.exit(1)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)

        from utils.metrics import collect_predictions, classification_metrics, regression_metrics
        labels, outputs = collect_predictions(test_loader, model, device)
        metrics = (
            classification_metrics(labels, outputs)
            if args.task == "toxicity"
            else regression_metrics(labels, outputs)
        )
        print(f"\nTest metrics: {metrics}")
        return

    # ---- Benchmark ----------------------------------------------------------
    if args.benchmark:
        ckpt_path = (
            args.checkpoint
            or os.path.join(args.output_dir, f"{run_name}_best.pt")
        )
        from experiments.evaluate import run_benchmark
        run_benchmark(
            root=args.data_dir,
            dataset_name=args.dataset,
            task=args.task,
            tox_target_col=args.tox_col,
            arch=args.arch,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            checkpoint_path=ckpt_path,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output_dir,
        )
        return

    # ---- Training (default) -------------------------------------------------
    from experiments.train import train
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task=args.task,
        epochs=args.epochs,
        lr=args.lr,
        pos_weight=args.pos_weight,
        early_stop_patience=args.early_stop,
        device=device,
        output_dir=args.output_dir,
        run_name=run_name,
    )

    logger.info(
        f"Best checkpoint: {results['checkpoint_path']}\n"
        f"Best val metrics: {results['best_val_metrics']}"
    )

    # Auto-plot training curves
    from utils.visualize import plot_training_curves
    history     = results["history"]
    train_losses = [h["train_loss"]  for h in history]
    val_losses   = [h["val_loss"]    for h in history]
    metric_key   = "val_auc_roc" if args.task == "toxicity" else "val_r2"
    val_metrics_hist = [h.get(metric_key, 0) for h in history]
    metric_label = "AUC-ROC" if args.task == "toxicity" else "R²"

    plot_training_curves(
        train_losses, val_losses, val_metrics_hist,
        metric_name=metric_label,
        save_path=os.path.join(args.output_dir, f"{run_name}_curves.png"),
    )


if __name__ == "__main__":
    main()
