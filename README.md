# BORF
BORF: Batch Ollivier Ricci Flow for unifying and addressing over-smoothing and over-squashing in GNN. 

## Requirements
To configure and activate the conda environment for this repository, run
```
conda env create -f environment.yml
conda activate borf 
pip install -r requirements.txt
```

## Experiments
### 1. For graph classification
To run experiments for the TUDataset benchmark, run the file ```run_graph_classification.py```. The following command will run the benchmark for BORF with 20 iterations:
```bash
python run_graph_classification.py --rewiring borf --num_iterations 20
```

To add options for number of edges added and removed for rewiring, add the --borf_batch_add and --borf_batch_remove options
```bash
# Runs BORF with 3 batches, add 3 edges per batch and remove 1 edge per batch
python run_graph_classification.py --rewiring borf --num_iterations 3 \
	--borf_batch_add 3 \
	--borf_batch_remove 1
```

### 2. For node classification
To run node classification, simply change the script name to `run_node_classification.py`. For example:
```bash
python run_node_classification.py --rewiring borf --num_iterations 3 \
	--borf_batch_add 3 \
	--borf_batch_remove 1
```

## Other rewiring methods
BORF is compared with other rewiring options, including SDRF and FoSR. The best hyper-paramters
for these methods are specified in the following scripts:
- For FoSR:
	- `scripts/run_node_fosr.sh`
	- `scripts/run_graph_fosr.sh`

- For SDRF:
	- `scripts/run_node_sdrf.sh`
	- `scripts/run_graph_sdrf.sh` 

### 1. Stochastic Discrete Ricci Flow (SDRF)
To run SDRF rewiring for both graph and node classification, add the rewiring option `sdrf_bfc` 
and the hyper-parameters for SDRF (`--sdrf_remove_edges` and `--num_iterations`). The following
is an example of running SDRF with 10 iterations and edge removal enabled:

```bash
python run_node_classification.py --layer_type GCN \
	--rewiring sdrf_bfc \
	--num_iterations 10 \
	--sdrf_remove_edges

python run_graph_classification.py --layer_type GCN \
	--rewiring sdrf_bfc \
	--num_iterations 10 \
	--sdrf_remove_edges
```

### 2. First-order Spectral Rewiring (FoSR)
To run FoSR, add the `fosr` rewiring option and the hyper-parameters for FoSR (`--num_iterations`).
The following is an example of running FoSR with 10 iterations:

```bash
python run_node_classification.py --layer_type GCN \
	--rewiring fosr \
	--num_iterations 10

python run_graph_classification.py --layer_type GCN \
	--rewiring fosr \
	--num_iterations 10
```

## Citation and reference
For technical details and full experiment results, please check [our paper](https://arxiv.org/abs/2211.15779).
```
@inproceedings{
nguyen2023revisiting,
title={Revisiting Over-smoothing and Over-squashing Using {Ollivier-Ricci} Curvature},
author={Khang Nguyen and Hieu Nong and Vinh Nguyen and Nhat Ho and Stanley Osher and Tan Nguyen},
booktitle={International Conference on Machine Learning},
year={2023}
}
```
