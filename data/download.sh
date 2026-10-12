#!/usr/bin/env bash
# MoleculeNet ESOL (Delaney 2004) and Tox21, from the DeepChem repository. Checksums pin the exact files used.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p raw
curl -sSL -o raw/delaney-processed.csv https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv
curl -sSL -o raw/tox21.csv.gz https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/tox21.csv.gz
sha256sum -c - <<SUMS
8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189  raw/delaney-processed.csv
48b34e4e59a2bae6f95734cd206c1869ab355f50a5e50bb7371a21f861d5e297  raw/tox21.csv.gz
SUMS
