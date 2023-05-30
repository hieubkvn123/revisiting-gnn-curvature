import torch
import torch.nn as nn
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, GatedGraphConv, GINConv, GATConv, FiLMConv, global_mean_pool
from torch_geometric.data import Data, InMemoryDataset

class SelfLoopGCNConv(torch.nn.Module):
    def __init__(self, in_features, out_features, args):
        super(SelfLoopGCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer1 = GCNConv(in_features, out_features)
        self.layer2 = GCNConv(in_features, out_features)
        self.device = args.device
    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        all_nodes = torch.arange(num_nodes)
        only_self_loops = torch.stack([all_nodes, all_nodes]).to(self.device)
        return self.layer1(x, edge_index) + self.layer2(x, only_self_loops)

class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.num_relations = args.num_relations
        self.layer_type = args.layer_type
        num_features = [args.input_dim] + list(args.hidden_layers) + [args.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            layers.append(self.get_layer(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=args.dropout)
        self.act_fn = ReLU()
    def get_layer(self, in_features, out_features):
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN" or self.layer_type == "Rewired-GCN-Sequential":
            return SelfLoopGCNConv(in_features, out_features, args=self.args)
        elif self.layer_type == "Rewired-GCN-Concurrent":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)))
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)
        elif self.layer_type == "GAT":
            return GATConv(in_features, out_features)
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph):
        x, edge_index, ptr, batch, root_mask = graph.x, graph.edge_index, graph.ptr, graph.batch, graph.root_mask
        x = x.float()
        batch_size = len(ptr) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != self.num_layers - 1:
                x = self.act_fn(x)
                x = self.dropout(x)

        return x[root_mask]
