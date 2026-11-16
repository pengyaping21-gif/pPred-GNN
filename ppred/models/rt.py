"""Original pPred RT architecture, retaining pretrained state-dict names."""
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GlobalAttention, Set2Set

from .gnn import GNN


class CustomLeakyReLU(nn.Module):
    def __init__(self, alpha=0.1, max_value=20.0):
        super(CustomLeakyReLU, self).__init__()
        self.alpha = alpha
        self.max_value = max_value
        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)

    def forward(self, x):
        x = self.leaky_relu(x)
        return torch.clamp(x, max=self.max_value)

class MolGraphPred(torch.nn.Module):
    """
    MolGraphPred: A PyTorch module for molecular graph prediction tasks.

    Args:
        num_layer (int): Number of GNN (Graph Neural Network) layers.
        emb_dim (int): Dimensionality of node embeddings.
        num_tasks (int): Number of prediction tasks.
        JK (str, optional): The strategy to combine node embeddings across GNN layers. Default is "last".
        drop_ratio (float, optional): Dropout ratio. Default is 0.
        graph_pooling (str, optional): The type of graph pooling operation. Default is "mean".
        gnn_type (str, optional): The type of GNN architecture. Default is "gin".
        device (str or torch.device, optional): Device to run the model on. Default is None.

    Raises:
        ValueError: If the number of GNN layers is less than 2.

    """

    def __init__(self, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin", device=None):
        super(MolGraphPred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.device = device
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(self.num_layer, self.emb_dim, self.device, self.JK, self.drop_ratio, gnn_type=gnn_type)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(
                    (self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(
                    gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1)
                                    * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(
                self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_embedding = nn.ModuleList([
                torch.nn.Linear(3 * self.mult * self.emb_dim, self.mult * self.emb_dim),
                torch.nn.Linear(self.mult * self.emb_dim, self.mult * self.emb_dim),
                torch.nn.Linear(self.mult * self.emb_dim, self.mult * self.emb_dim),
                torch.nn.Linear(self.mult * self.emb_dim, self.mult * self.emb_dim)
            ])
            self.graph_pred_linear = torch.nn.Linear(
                self.mult * self.emb_dim, self.num_tasks)
        self.graph_pred_activation = torch.nn.ReLU()

        self.fp_fc1=nn.Linear(1489, 512)
        self.fp_act_func = nn.ReLU()
        self.fp_fc2 = nn.Linear(512, 300)
        self.fp_dropout = nn.Dropout(p=0.0)

        self.conv_net = nn.Sequential(

            nn.Conv1d(in_channels=118, out_channels=256, kernel_size=8, padding='same'),
            CustomLeakyReLU(alpha=0.1, max_value=20.0),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=8, padding='same'),
            CustomLeakyReLU(alpha=0.1, max_value=20.0),
            nn.MaxPool1d(kernel_size=2, stride=2),

        )

        self.conv_net1 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=8, padding='same'),
            CustomLeakyReLU(alpha=0.1, max_value=20.0),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, padding='same'),
            CustomLeakyReLU(alpha=0.1, max_value=20.0),
            nn.MaxPool1d(kernel_size=2, stride=2),

        )

        self.conv_net2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, padding='same'),
            CustomLeakyReLU(alpha=0.1, max_value=20.0),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, padding='same'),
            CustomLeakyReLU(alpha=0.1, max_value=20.0),
            nn.MaxPool1d(kernel_size=2, stride=2),

        )
        self.flatten = nn.Flatten()

        self.conv_linear = nn.Linear(in_features=448, out_features=300, bias=True)


    def forward(self, *argv):
        if len(argv) == 1:
            data = argv[0]
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            batch = data.batch

            fp = data.fp
            fp = fp.reshape(-1, 1489)
            fpn_out = self.fp_fc1(fp.float())
            fpn_out = self.fp_dropout(fpn_out)
            fpn_out = self.fp_act_func(fpn_out)
            fpn_out = self.fp_fc2(fpn_out)

            node_representation = self.gnn(x, edge_index, edge_attr)
            node_pool = self.pool(node_representation, batch)

            composition_x = data.element_composition_array.reshape(-1, 60, 118)
            composition_x = composition_x.transpose(dim0=1, dim1=2).float()
            composition_x = self.conv_net(composition_x)
            composition_x = self.conv_net1(composition_x)
            composition_x = self.conv_net2(composition_x)
            composition_x = self.flatten(composition_x)
            composition_x = self.conv_linear(composition_x)

            combined_node_pool = torch.cat((node_pool, fpn_out), dim=1)
            combined_node_pool = torch.cat((combined_node_pool, composition_x), dim=1)

            for index, module in enumerate(self.graph_embedding):
                if index == 0:
                    node_pool = module(combined_node_pool)
                else:
                    node_pool = module(node_pool)

            gnn_out_cls_result = self.graph_pred_linear(node_pool)
            cls_result1 = gnn_out_cls_result

            node_representation = None
            return cls_result1, node_representation


        else:
            raise ValueError("unmatched number of arguments.")

