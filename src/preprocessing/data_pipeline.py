import numpy as np
import trimesh
import torch
from torch_geometric.data import Data
import scipy.sparse as sp

from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
import os

class DataPipeline:
    def __init__(self, data_root_path, save_path):
        # /Users/samuelandersson/Dev/CodeSprint/GeometricDL-for-CAD-models/data/raw/ModelNet10
        # '/Users/samuelandersson/Dev/CodeSprint/GeometricDL-for-CAD-models/data/processed'
        self.data_root_path = data_root_path
        self.save_path = save_path

    def load_mesh_from_file(self, file_path):
        # Load the mesh from the OFF file
        mesh = trimesh.load_mesh(file_path)
        return mesh
    
    def compute_coo_matrix_vectorised(self, vertices: np.ndarray, faces: np.ndarray) -> sp.coo_matrix:
        """Compute a sparse adjacency matrix from the given vertices and faces.

        Args:
            vertices (np.ndarray): A numpy array of vertices.
            faces (np.ndarray): A numpy array of faces.

        Returns:
            sp.coo_matrix: A sparse adjacency matrix in COO format.
        """
        # Create arrays representing all possible edge pairs for each face
        rows = np.hstack([faces[:, i] for i in range(3)] * 2)
        cols = np.hstack([faces[:, (i + 1) % 3] for i in range(3)] + [faces[:, (i + 2) % 3] for i in range(3)])

        # Create a sparse matrix in COO format
        data = np.ones(len(rows))  # The values for non-zero elements
        return sp.coo_matrix((data, (rows, cols)), shape=(len(vertices), len(vertices)))

    
    def coo_to_pyg_data(self, coo_adjacency_matrix, node_features):
        # Use the row and column attributes of the COO matrix to create the edge index tensor directly
        edge_index = torch.tensor(np.vstack([coo_adjacency_matrix.row, coo_adjacency_matrix.col]), dtype=torch.long)

        # Convert node features to a tensor feature
        x = torch.tensor(node_features, dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index)

    def process_one_file_coo_vectorised(self, file_path):
        mesh = self.load_mesh_from_file(file_path)
        
        # Get vertices and faces
        vertices = mesh.vertices
        vertex_normals = mesh.vertex_normals  # Extract vertex normals

        # Concatenate vertices and vertex_normals to create node features
        node_features = np.hstack([vertices, vertex_normals])

        faces = mesh.faces
        coo_adjacency_matrix = self.compute_coo_matrix_vectorised(vertices, faces)

        pyg_data = self.coo_to_pyg_data(coo_adjacency_matrix, node_features)

        return pyg_data


    def process_dataset(self):
        """Load the entire dataset from the given directory.

        Args:
            root_path (str): The root directory of the dataset.

        Returns:
            list: A list of PyTorch Geometric data objects.
        """
        dataset = []
        categories = sorted([c for c in os.listdir(self.data_root_path) if os.path.isdir(os.path.join(self.data_root_path, c))])
        category_to_label = {category: label for label, category in enumerate(categories)}

        print("Starting to load dataset...")

        for category_idx, category in enumerate(categories):
            print(f"Processing category {category} ({category_idx + 1}/{len(categories)})...")
            for subset in ['train', 'test']:
                folder_path = os.path.join(self.data_root_path, category, subset)
                if os.path.isdir(folder_path):
                    file_names = [f for f in os.listdir(folder_path) if f.endswith('.off')]
                    for file_idx, file_name in enumerate(file_names):
                        file_path = os.path.join(folder_path, file_name)
                        pyg_data = self.process_one_file_coo_vectorised(file_path)
                        pyg_data.y = torch.tensor(category_to_label[category], dtype=torch.long).unsqueeze(0)  # Add label
                        dataset.append(pyg_data)
                        if (file_idx + 1) % 50 == 0:
                            print(f"  Loaded {file_idx + 1}/{len(file_names)} files from {subset} subset...")

        print("Loading complete!")
        return dataset



    def split_dataset(self, dataset, test_size=0.2):
        """Split the dataset into training and testing sets.

        Args:
            dataset (list): A list of PyTorch Geometric data objects.
            test_size (float): The proportion of the dataset to include in the test split.

        Returns:
            tuple: Train and test datasets.
        """
        return train_test_split(dataset, test_size=test_size, random_state=42)

    def create_dataloaders(self, train_data, test_data, batch_size=32):
        """Create PyTorch Geometric DataLoaders for train and test datasets.

        Args:
            train_data (list): Training dataset.
            test_data (list): Testing dataset.
            batch_size (int): Number of samples per batch.

        Returns:
            tuple: Train and test DataLoaders.
        """
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)
        return train_loader, test_loader

    # def save_dataloader(self, dataloader, file_name):
    #     """Save the DataLoader to a file.

    #     Args:
    #         dataloader (DataLoader): The DataLoader to save.
    #         file_path (str): The path to the file.
    #     """
    #     # torch.save(dataloader, self.save_path)
    #     torch.save(dataloader, f'{self.save_path}/{file_name}')

    def save_dataset(self, dataset, file_name):
        """Save the PyTorch Geometric dataset to a file.

        Args:
            dataset (torch_geometric.data.Dataset): The dataset to save.
            file_name (str): The name of the file.
        """
        torch.save(dataset, f'{self.save_path}\{file_name}')

    def load_dataset(self, file_name):
        """Load a PyTorch Geometric dataset from a file.

        Args:
            file_name (str): The name of the file.

        Returns:
            torch_geometric.data.Dataset: The loaded dataset.
        """
        return torch.load(f'{self.save_path}\{file_name}')

    def load_dataloader(self, file_name):
        """Load the DataLoader from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            DataLoader: The loaded DataLoader.
        """
        dataset = self.load_dataset(file_name)

        return DataLoader(dataset, batch_size=10, shuffle=True)
    