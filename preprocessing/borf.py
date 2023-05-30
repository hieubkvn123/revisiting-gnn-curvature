import os
import ot
import time
import torch
import pathlib
import numpy as np
import pandas as pd
import multiprocessing as mp
import networkx as nx
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
)
from torch_geometric.datasets import TUDataset
from GraphRicciCurvature.OllivierRicci import OllivierRicci

class CurvaturePlainGraph():
    def __init__(self, G, device=None):
        self.G = G
        self.V = len(G.nodes)
        self.E = list(G.edges)
        self.adjacency_matrix = np.full((self.V,self.V),np.inf)
        self.dist = self.adjacency_matrix.copy()

        if(device is None):
          self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
          self.device = device
        
        for index in range(self.V):
            self.adjacency_matrix[index, index] = 0
        for index, edge in enumerate(self.E):
            self.adjacency_matrix[edge[0], edge[1]] = 1
            self.adjacency_matrix[edge[1], edge[0]] = 1
        
        # Floyd Warshall
        self.dist = self._floyd_warshall()

    def __str__(self):
        return f'The graph contains {self.V} nodes and {len(self.E)} edges {self.E}. '

    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.E)
        nx.draw_networkx(G)
        plt.show()

    def _dijkstra(self):
        for i in range(self.V):
            for j in range(self.V):
                try:
                    self.dist[i][j] = len(nx.dijkstra_path(self.G, i, j))
                except nx.NetworkXNoPath:
                    continue
        return self.dist

    def _floyd_warshall(self):
        self.dist = self.adjacency_matrix.copy()
        for k in range(self.V):
            for i in range(self.V):
                for j in range(self.V):
                    self.dist[i][j] = min(self.dist[i][j], self.dist[i][k] + self.dist[k][j])
        return self.dist

    def _to_tensor(self, x):
        x = torch.Tensor(x).to(self.device)
        return x

    def _to_numpy(self, x):
        if(torch.cuda.is_available()):
            return x.cpu().detach().numpy()
        return x.detach().numpy()

    def _transport_plan_uv(self, u, v, method = 'OTD', u_neighbors=None, v_neighbors=None):
        u_neighbors = [p for p in range(self.V) if self.adjacency_matrix[u][p] == 1] if u_neighbors is None else u_neighbors
        v_neighbors = [q for q in range(self.V) if self.adjacency_matrix[v][q] == 1] if v_neighbors is None else v_neighbors
        u_deg = len(u_neighbors)
        v_deg = len(v_neighbors)

        # Instead of using fractions [1/n,...,1/n], [1/m,...,1/m], we use [m,...,m], [n,...,n] and then divides by mn
        mu = self._to_tensor(np.full(u_deg, v_deg))
        mv = self._to_tensor(np.full(v_deg, u_deg))
        sub_indices = np.ix_(u_neighbors, v_neighbors)
        dist_matrix = self._to_tensor(self.dist[sub_indices])
        dist_matrix[dist_matrix == np.inf] = 0 # Correct the dist matrix
        
        # Update distance matrix
        self.d = dist_matrix
        if method == 'OTD':
            optimal_plan = self._to_numpy(ot.emd(mu, mv, dist_matrix))
        else:
            raise NotImplemented
        optimal_plan = optimal_plan/(u_deg*v_deg) # PI
        optimal_cost = optimal_plan*self._to_numpy(dist_matrix)
        optimal_total_cost = np.sum(optimal_cost)
        optimal_cost = pd.DataFrame(optimal_cost, columns=v_neighbors, index=u_neighbors)
        return optimal_total_cost, optimal_cost

    def add_edge(self, p, q, inter_up, inter_vq):
        # TODO : Need to replace with a more efficient algorithm
        self.adjacency_matrix[p, q] = 1
        self.adjacency_matrix[q, p] = 1
        
        # self.dist = self._floyd_warshall()
        self.dist[p, q] = 1
        self.dist[q, p] = 1

        # Add edge to edge list
        self.E.append((p, q))

        for k in inter_up:
            self.dist[k, q] = min(2, self.dist[k, q])
            
        for l in inter_vq:
            self.dist[l, p] = min(2, self.dist[l, p])

    def remove_edge(self, i, j):
        self.adjacency_matrix[i, j] = 0
        self.adjacency_matrix[j, i] = 0
        # self.dist = self._floyd_warshall()
        self.dist[i, j] = np.inf
        self.dist[j, i] = np.inf

        self.E.remove((i, j))

    def curvature_uv(self, u, v, method = 'OTD', u_neighbors=None, v_neighbors=None):
        optimal_cost, optimal_plan = self._transport_plan_uv(u, v, method, u_neighbors=u_neighbors, v_neighbors=v_neighbors)
        return 1 - optimal_cost/self.dist[u,v], optimal_plan

    def edge_curvatures(self, method = 'OTD', return_transport_cost=False):
        edge_curvature_dict = {}
        transport_plan_dict = {}
        for edge in self.E:
            edge_curvature_dict[edge], transport_plan_dict[edge] = self.curvature_uv(edge[0], edge[1], method)

        if(return_transport_cost):
            return edge_curvature_dict, transport_plan_dict
        return edge_curvature_dict

    def all_curvatures(self, method = 'OTD'):
        C = np.zeros((self.V, self.V))
        for u in range(self.V):
            for v in range(u+1, self.V):
                C[u,v] = self.curvature_uv(u,v,method)
        C = C + np.transpose(C) + np.eye(self.V)
        C = np.hstack((np.reshape([str(u) for u in np.arange(self.V)],(self.V,1)), C))
        head = ['C'] + [str(u) for u in range(self.V)]
        print(tabulate(C, floatfmt=".2f", headers=head, tablefmt="presto"))

def _softmax(a, tau=1):
    exp_a = np.exp(a * tau)
    return exp_a / exp_a.sum()

def _preprocess_data(data, is_undirected=False):
    # Get necessary data information
    N = data.x.shape[0]
    m = data.edge_index.shape[1]

    # Compute the adjacency matrix
    if not "edge_type" in data.keys:
        edge_type = np.zeros(m, dtype=int)
    else:
        edge_type = data.edge_type

    # Convert graph to Networkx
    G = to_networkx(data)
    if is_undirected:
        G = G.to_undirected()

    return G, N, edge_type

def _get_neighbors(x, G, is_undirected=False, is_source=False):
    if is_undirected:
        x_neighbors = list(G.neighbors(x)) #+ [x]
    else:
        if(is_source):
          x_neighbors = list(G.successors(x)) #+ [x]
        else:
          x_neighbors = list(G.predecessors(x)) #+ [x]
    return x_neighbors

def _get_rewire_candidates(G, x_neighbors, y_neighbors):
    candidates = []
    for i in x_neighbors:
        for j in y_neighbors:
            if (i != j) and (not G.has_edge(i, j)):
                candidates.append((i, j))
    return candidates

def _calculate_improvement(graph, C, x, y, x_neighbors, y_neighbors, k, l):
    """
    Calculate the curvature performance of x -> y when k -> l is added.
    """
    graph.add_edge(k, l)
    old_curvature = C[(x, y)]

    new_curvature, _ = graph.curvature_uv(x, y, u_neighbors=x_neighbors, v_neighbors=y_neighbors)
    improvement = new_curvature - old_curvature
    graph.remove_edge(k, l)

    return new_curvature, old_curvature

def brf2(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    batch_add=4,
    batch_remove=2,
    device=None
):
    # Preprocess data
    G, N, edge_type = _preprocess_data(data)

    # Rewiring begins
    for _ in range(loops):
        # Compute ORC
        graph = CurvaturePlainGraph(G, device=device)
        C, PI = graph.edge_curvatures(method='OTD', return_transport_cost=True)
        _C = sorted(C, key=C.get)

        # Get top negative and positive curved edges
        most_pos_edges = _C[-batch_remove:]
        most_neg_edges = _C[:batch_add]

        # Add edges
        for (u, v) in most_neg_edges:
            pi = PI[(u, v)]
            p, q = np.unravel_index(pi.values.argmax(), pi.values.shape)
            p, q = pi.index[p], pi.columns[q]
            
            if(p != q and not G.has_edge(p, q)):
                G.add_edge(p, q)

        # Remove edges
        for (u, v) in most_pos_edges:
            if(G.has_edge(u, v)):
                G.remove_edge(u, v)
    return from_networkx(G).edge_index, torch.tensor(edge_type)


def borf3(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    batch_add=4,
    batch_remove=2,
    device=None,
    save_dir='rewired_graphs',
    dataset_name=None,
    graph_index=0,
    debug=False
):
    # Check if there is a preprocessed graph
    dirname = f'{save_dir}/{dataset_name}'
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    edge_index_filename = os.path.join(dirname, f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_edge_index_{graph_index}.pt')
    edge_type_filename = os.path.join(dirname, f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_edge_type_{graph_index}.pt')

    if(os.path.exists(edge_index_filename) and os.path.exists(edge_type_filename)):
        if(debug) : print(f'[INFO] Rewired graph for {loops} iterations, {batch_add} edge additions and {batch_remove} edge removal exists...')
        with open(edge_index_filename, 'rb') as f:
            edge_index = torch.load(f)
        with open(edge_type_filename, 'rb') as f:
            edge_type = torch.load(f)
        return edge_index, edge_type

    # Preprocess data
    G, N, edge_type = _preprocess_data(data)

    # Rewiring begins
    for _ in range(loops):
        # Compute ORC
        orc = OllivierRicci(G, alpha=0)
        orc.compute_ricci_curvature()
        _C = sorted(orc.G.edges, key=lambda x: orc.G[x[0]][x[1]]['ricciCurvature']['rc_curvature'])

        # Get top negative and positive curved edges
        most_pos_edges = _C[-batch_remove:]
        most_neg_edges = _C[:batch_add]

        # Add edges
        for (u, v) in most_neg_edges:
            pi = orc.G[u][v]['ricciCurvature']['rc_transport_cost']
            p, q = np.unravel_index(pi.values.argmax(), pi.values.shape)
            p, q = pi.index[p], pi.columns[q]
            
            if(p != q and not G.has_edge(p, q)):
                G.add_edge(p, q)

        # Remove edges
        for (u, v) in most_pos_edges:
            if(G.has_edge(u, v)):
                G.remove_edge(u, v)

    edge_index = from_networkx(G).edge_index
    edge_type = torch.zeros(size=(len(G.edges),)).type(torch.LongTensor)
    # edge_type = torch.tensor(edge_type)

    if(debug) : print(f'[INFO] Saving edge_index to {edge_index_filename}')
    with open(edge_index_filename, 'wb') as f:
        torch.save(edge_index, f)

    if(debug) : print(f'[INFO] Saving edge_type to {edge_type_filename}')
    with open(edge_type_filename, 'wb') as f:
        torch.save(edge_type, f)

    return edge_index, edge_type
