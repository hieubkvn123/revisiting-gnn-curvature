import os.path
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from preprocessing import rewiring, sdrf, fosr, digl
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj
from torch_geometric.datasets import TUDataset
import time

font = {'size': 16}
matplotlib.rc('font', **font)

def average_spectral_gap(dataset):
    # computes the average spectral gap out of all graphs in a dataset
    spectral_gaps = []
    for graph in dataset:
        G = to_networkx(graph, to_undirected=True)
        spectral_gap = rewiring.spectral_gap(G)
        spectral_gaps.append(spectral_gap)
    return sum(spectral_gaps) / len(spectral_gaps)

if not os.path.isfile("results/time_complexity.csv"):

    mutag = list(TUDataset(root="data", name="MUTAG"))
    enzymes = list(TUDataset(root="data", name="ENZYMES"))
    proteins = list(TUDataset(root="data", name="PROTEINS"))
    collab = list(TUDataset(root="data", name="COLLAB"))
    imdb = list(TUDataset(root="data", name="IMDB-BINARY"))
    reddit = list(TUDataset(root="data", name="REDDIT-BINARY"))
    datasets = {"reddit": reddit, "imdb": imdb, "mutag": mutag, "enzymes": enzymes, "proteins": proteins, "collab": collab}

    for key in datasets:
        if key in ["reddit", "imdb", "collab"]:
            for graph in datasets[key]:
                n = graph.num_nodes
                graph.x = torch.ones((n,1))

    num_iterations = 10

    for key in ["reddit", "reddit"]:
        for rewiring_method in ["fosr"]:
            print(key, rewiring_method)
            dataset = datasets[key]
            time_start = time.time()
            if rewiring_method == "sdrf":
                for i in range(len(dataset)):
                    # add an edge to each graph once
                    dataset[i].edge_index, dataset[i].edge_type = sdrf.sdrf(dataset[i], loops=num_iterations, remove_edges=False, is_undirected=True)
            elif rewiring_method == "fosr":
                for i in range(len(dataset)):
                    edge_index, edge_type, _ = fosr.edge_rewire(dataset[i].edge_index.numpy(), num_iterations=num_iterations)
                    dataset[i].edge_index = torch.tensor(edge_index)
                    dataset[i].edge_type = torch.tensor(edge_type)
            time_end = time.time()
            print("Time taken: ", time_end - time_start)
