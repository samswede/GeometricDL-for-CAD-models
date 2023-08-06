import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear, BatchNorm1d
from torch_geometric.data import Batch

class AdvancedGNN(torch.nn.Module):
    """
    Advanced Graph Neural Network (GNN) architecture for graph classification.

    Args:
        num_node_features (int): Number of features for each node in the graph.
        num_classes (int): Number of classes for classification.
        hidden_dim (int, optional): Hidden dimension size for the GAT layers. Default is 64.
        num_heads (int, optional): Number of attention heads for the GAT layers. Default is 4.
        dropout (float, optional): Dropout rate for the dropout layers. Default is 0.1.
    """

    def __init__(self, num_node_features: int, num_classes: int, hidden_dim: int = 64,
                 num_heads: int = 4, dropout: float = 0.1):
        super(AdvancedGNN, self).__init__()

        # GAT layers
        self.gat1 = GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)

        # Batch Normalization layers
        self.bn1 = BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = BatchNorm1d(hidden_dim * num_heads)

        # MLP layers
        self.fc1 = Linear(hidden_dim * num_heads, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)

        # Dropout
        self.dropout = dropout

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass for the advanced GNN model.

        Args:
            data (Batch): Input data containing node features, edge indices, and batch information.

        Returns:
            torch.Tensor: Output tensor representing the log probabilities for each class.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GAT layers with residual connections and batch normalization
        x_res = x
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.bn1(x + x_res)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_res = x
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.bn2(x + x_res)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling (mean pooling)
        x = global_mean_pool(x, batch)

        # Apply MLP layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
