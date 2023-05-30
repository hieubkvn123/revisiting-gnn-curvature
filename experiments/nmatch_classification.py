import torch
import numpy as np
from attrdict import AttrDict
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from math import inf

from models.nmatch_model import GCN

default_args = AttrDict(
    {"learning_rate": 1e-3,
    "max_epochs": 1000000,
    "display": True,
    "device": None,
    "eval_every": 1,
    "stopping_criterion": "validation",
    "stopping_threshold": 1.01,
    "patience": 5,
    "train_fraction": 0.9,
    "validation_fraction": 0.05,
    "test_fraction": 0.05,
    "dropout": 0.0,
    "weight_decay": 1e-5,
    "input_dim": None,
    "hidden_dim": 32,
    "output_dim": 1,
    "hidden_layers": None,
    "num_layers": 1,
    "batch_size": 64,
    "layer_type": "GCN",
    "num_relations": 1
    }
    )

class Experiment:
    def __init__(self, args=None, dataset=None):
        self.args = default_args + args
        self.dataset = dataset
        self.loss_fn = torch.nn.CrossEntropyLoss()

        if self.args.device is None:
            self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.args.hidden_layers is None:
            self.args.hidden_layers = [self.args.hidden_dim] * self.args.num_layers
        if self.args.input_dim is None:
            self.args.input_dim = self.dataset[0].x.shape[1]
        self.args.output_dim = max([elt.y for elt in self.dataset]) + 1
        self.model = GCN(self.args).to(self.args.device)
        
    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer)

        if self.args.display:
            print("Starting training")
        best_train_acc = 0
        goal_train_acc = 0
        best_epoch = 0
        epochs_no_improve = 0
        train_size = len(self.dataset)

        train_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)

        for epoch in range(self.args.max_epochs):
            self.model.train()
            sample_size = 0
            optimizer.zero_grad()

            for graph in train_loader:
                graph = graph.to(self.args.device)
                y = graph.y.to(self.args.device)

                out = self.model(graph)
                loss = self.loss_fn(input=out, target=y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            new_best_str = ''

            if epoch % self.args.eval_every == 0:
                train_acc = self.eval(train_loader)
                scheduler.step(loss)


                if train_acc > goal_train_acc:
                    best_train_acc = train_acc
                    goal_train_acc = best_train_acc * self.args.stopping_threshold
                    epochs_no_improve = 0
                    new_best_str = ' (new best train)'
                elif train_acc > best_train_acc:
                    best_train_acc = train_acc
                    epochs_no_improve += 1
                else:
                    epochs_no_improve += 1
                if self.args.display:
                    print(f'Epoch {epoch}, Train accuracy: {train_acc}')
                if epochs_no_improve > self.args.patience:
                    if self.args.display:
                        print(f'{self.args.patience} epochs without improvement, stopping training')
                        print(f'Best train accuracy: {best_train_acc}')
                    return best_train_acc

    def eval(self, loader):
        self.model.eval()
        sample_size = len(loader.dataset)
        with torch.no_grad():
            total_correct = 0
            for graph in loader:
                graph = graph.to(self.args.device)
                y = graph.y.to(self.args.device)
                out = self.model(graph)
                guess = torch.argmax(out, dim=1)
                acc = sum(y == guess)
                total_correct += acc
                
        return total_correct / sample_size