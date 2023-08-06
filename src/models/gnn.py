import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear
from torch_geometric.data import Batch

class GNN(torch.nn.Module):
    """
    Graph Neural Network with Multi-Layer Perceptron (GNN-MLP) for graph classification.

    Args:
        num_node_features (int): Number of features for each node in the graph.
        num_classes (int): Number of classes for classification.
        hidden_dim (int, optional): Hidden dimension size for the GNN and MLP layers. Default is 64.
    """

    def __init__(self, num_node_features: int, num_classes: int, hidden_dim: int = 64):
        super(GNN, self).__init__()

        # GNN layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # MLP layers
        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass for the GNN-MLP model.

        Args:
            data (Batch): Input data containing node features, edge indices, and batch information.

        Returns:
            torch.Tensor: Output tensor representing the log probabilities for each class.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GNN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global pooling (mean pooling)
        x = global_mean_pool(x, batch)

        # Apply MLP layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
