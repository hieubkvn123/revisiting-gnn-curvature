import torch
import numpy as np
from measure_smoothing import dirichlet_normalized
from attrdict import AttrDict
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from math import inf

from models.graph_model import GNN

default_args = AttrDict(
    {"learning_rate": 1e-3,
    "max_epochs": 1000000,
    "display": True,
    "device": None,
    "eval_every": 1,
    "stopping_criterion": "validation",
    "stopping_threshold": 1.01,
    "patience": 20,
    "train_fraction": 0.8,
    "validation_fraction": 0.1,
    "test_fraction": 0.1,
    "dropout": 0.0,
    "weight_decay": 1e-5,
    "input_dim": None,
    "hidden_dim": 32,
    "output_dim": 1,
    "hidden_layers": None,
    "num_layers": 1,
    "batch_size": 64,
    "layer_type": "R-GCN",
    "num_relations": 2,
    "last_layer_fa": False
    }
    )

class Experiment:
    def __init__(self, args=None, dataset=None, train_dataset=None, validation_dataset=None, test_dataset=None):
        self.args = default_args + args
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.loss_fn = torch.nn.CrossEntropyLoss()

        if self.args.device is None:
            self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.args.hidden_layers is None:
            self.args.hidden_layers = [self.args.hidden_dim] * self.args.num_layers
        if self.args.input_dim is None:
            self.args.input_dim = self.dataset[0].x.shape[1]
        for graph in self.dataset:
            if not "edge_type" in graph.keys:
                num_edges = graph.edge_index.shape[1]
                graph.edge_type = torch.zeros(num_edges, dtype=int)
        if self.args.num_relations is None:
            if self.args.rewiring == "None":
                self.args.num_relations = 1
            else:
                self.args.num_relations = 2
        self.model = GNN(self.args).to(self.args.device)
       
        if self.test_dataset is None:
            dataset_size = len(self.dataset)
            train_size = int(self.args.train_fraction * dataset_size)
            validation_size = int(self.args.validation_fraction * dataset_size)
            test_size = dataset_size - train_size - validation_size
            self.train_dataset, self.validation_dataset, self.test_dataset = random_split(self.dataset,[train_size, validation_size, test_size])
        elif self.validation_dataset is None:
            train_size = int(self.args.train_fraction * len(self.train_dataset))
            validation_size = len(self.args.train_data) - train_size
            self.args.train_data, self.args.validation_data = random_split(self.args.train_data, [train_size, validation_size])
        
    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer)

        best_validation_acc = 0.0
        best_train_acc = 0.0
        best_test_acc = 0.0
        train_goal = 0.0
        validation_goal = 0.0
        epochs_no_improve = 0

        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_dataset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True)
        complete_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)

        for epoch in range(1, 1 + self.args.max_epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()

            for graph in train_loader:
                graph = graph.to(self.args.device)
                y = graph.y.to(self.args.device)

                out = self.model(graph)
                loss = self.loss_fn(input=out, target=y)
                total_loss += loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            new_best_str = ''
            scheduler.step(total_loss)
            if epoch % self.args.eval_every == 0:
                train_acc = self.eval(loader=train_loader)
                validation_acc = self.eval(loader=validation_loader)
                test_acc = self.eval(loader=test_loader)

                if self.args.stopping_criterion == "train":
                    if train_acc > train_goal:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve = 0
                        train_goal = train_acc * self.args.stopping_threshold
                        new_best_str = ' (new best train)'
                    elif train_acc > best_train_acc:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve += 1
                    else:
                        epochs_no_improve += 1
                elif self.args.stopping_criterion == 'validation':
                    if validation_acc > validation_goal:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve = 0
                        validation_goal = validation_acc * self.args.stopping_threshold
                        new_best_str = ' (new best validation)'
                    elif validation_acc > best_validation_acc:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve += 1
                    else:
                        epochs_no_improve += 1
                if self.args.display:
                    print(f'Epoch {epoch}, Train acc: {train_acc}, Validation acc: {validation_acc}{new_best_str}, Test acc: {test_acc}')
                if epochs_no_improve > self.args.patience:
                    if self.args.display:
                        print(f'{self.args.patience} epochs without improvement, stopping training')
                        print(f'Best train acc: {best_train_acc}, Best validation acc: {best_validation_acc}, Best test acc: {best_test_acc}')
                        energy = 0
                    return best_train_acc, best_validation_acc, best_test_acc, energy
        if self.args.display:
            print('Reached max epoch count, stopping training')
            print(f'Best train acc: {best_train_acc}, Best validation acc: {best_validation_acc}, Best test acc: {best_test_acc}')

        energy = 0
        return best_train_acc, best_validation_acc, best_test_acc, energy

    def eval(self, loader):
        self.model.eval()
        sample_size = len(loader.dataset)
        with torch.no_grad():
            total_correct = 0
            for graph in loader:
                graph = graph.to(self.args.device)
                y = graph.y.to(self.args.device)
                out = self.model(graph)
                _, pred = out.max(dim=1)
                total_correct += pred.eq(y).sum().item()
                
        return total_correct / sample_size
    def check_dirichlet(self, loader):
        self.model.eval()
        sample_size = len(loader.dataset)
        with torch.no_grad():
            total_energy = 0
            for graph in loader:
                graph = graph.to(self.args.device)
                total_energy += self.model(graph, measure_dirichlet=True)
        return total_energy / sample_size
