import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGPooling, global_mean_pool
from torch.nn import Linear
from torch_geometric.data import Batch

class SagPoolGAT(torch.nn.Module):
    """
    Scalable Graph Neural Network with Graph Attention Layers (GAT), Self-Attention Graph Pooling (SAGPooling),
    and Multi-Layer Perceptron (MLP) for graph classification.

    Args:
        num_node_features (int): Number of features for each node in the graph.
        num_classes (int): Number of classes for classification.
        hidden_dim (int, optional): Hidden dimension size for the GNN and MLP layers. Default is 64.
        heads (int, optional): Number of attention heads for GAT layers. Default is 4.
        ratio (float, optional): Ratio of nodes to keep after pooling. Default is 0.5.
    """

    def __init__(self, num_node_features: int, num_classes: int, hidden_dim: int = 64, heads: int = 4, ratio: float = 0.5):
        super(SagPoolGAT, self).__init__()

        # GNN layers with Graph Attention mechanism
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=heads, concat=True)
        self.pool1 = SAGPooling(hidden_dim * heads, ratio)
        self.conv2 = GATConv(heads * hidden_dim, hidden_dim, heads=heads, concat=True)
        self.pool2 = SAGPooling(hidden_dim * heads, ratio)

        # MLP layers
        self.fc1 = Linear(heads * hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass for the Scalable GNN model.

        Args:
            data (Batch): Input data containing node features, edge indices, and batch information.

        Returns:
            torch.Tensor: Output tensor representing the log probabilities for each class.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GNN layers with Graph Attention mechanism
        x = self.conv1(x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        x = F.relu(x)

        # Global pooling (mean pooling)
        x = global_mean_pool(x, batch)

        # Apply MLP layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
