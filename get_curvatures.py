from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import WebKB, WikipediaNetwork, Planetoid
from torch_geometric.utils import to_networkx, from_networkx, to_undirected
from GraphRicciCurvature.OllivierRicci import OllivierRicci

import pathlib
import warnings
import numpy as np 
import pandas as pd
warnings.filterwarnings('ignore')

### Node classification ###
cornell = WebKB(root="data", name="Cornell")
wisconsin = WebKB(root="data", name="Wisconsin")
texas = WebKB(root="data", name="Texas")
chameleon = WikipediaNetwork(root="data", name="chameleon")
cora = Planetoid(root="data", name="cora")
citeseer = Planetoid(root="data", name="citeseer")

### Graph classification ###
mutag = list(TUDataset(root="data", name="MUTAG"))
enzymes = list(TUDataset(root="data", name="ENZYMES"))
proteins = list(TUDataset(root="data", name="PROTEINS"))
imdb = list(TUDataset(root="data", name="IMDB-BINARY"))

### Datasets ###
datasets = {
    'node_cls' : {
        'cornell' : cornell,
        'wisconsin' : wisconsin,
        'texas' : texas,
        'chameleon' : chameleon,
        'cora' : cora,
        'citeseer' : citeseer
    },
    'graph_cls' : {
        'mutag' : mutag,
        'enzymes' : enzymes,
        'proteins' : proteins,
        'imdb' : imdb
    }
}

def _get_single_graph_statistics(G):
    orc = OllivierRicci(G, alpha=0)
    orc.compute_ricci_curvature()

    all_curvatures = []
    for i, j in orc.G.edges:
        all_curvatures.append(orc.G[i][j]['ricciCurvature']['rc_curvature'])
    all_curvatures = np.array(all_curvatures)

    return all_curvatures.mean(), all_curvatures.std()

### Calculate statistics for node classification ###
data_statistics = {}
for key in datasets['node_cls']:
    print(f'[INFO] Calculating curvatures for {key}')
    dataset = datasets['node_cls'][key]
    G = to_networkx(dataset.data)
    mean, std = _get_single_graph_statistics(G)

    data_statistics[key] = {
        'mean' : mean, 
        'std' : std
    }

for key in datasets['graph_cls']:
    print(f'[INFO] Calculating curvatures for {key}')
    dataset = datasets['graph_cls'][key]
    avg_curvatures = []
    for i in range(len(dataset)):
        G = to_networkx(dataset[i])
        mean, _ = _get_single_graph_statistics(G)
        avg_curvatures.append(mean)
    avg_curvatures = np.array(avg_curvatures)
    data_statistics[key] = {
        'mean' : avg_curvatures.mean(),
        'std' : avg_curvatures.std()
    }


# Display curvatures
df = pd.DataFrame(data=data_statistics)
print(df)

# Save curvatures
print('[INFO] Saving curvatures to results/curvatures.csv')
pathlib.Path('results').mkdir(parents=True, exist_ok=True)
df.to_csv('results/curvatures.csv')

