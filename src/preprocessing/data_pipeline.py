
import trimesh
import torch
from torch_geometric.data import Data
import scipy.sparse as sp

class DataPipeline:
    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path

    def load_mesh_from_file(self, file_path):
        # Load the mesh from the OFF file
        mesh = trimesh.load_mesh(file_path)
        return mesh
    
    def compute_sparse_adjacency_matrix(self, mesh):
        # Get vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Create an adjacency matrix
        sparse_adjacency_matrix = sp.csr_matrix((len(vertices), len(vertices)))
        for face in faces:
            for i in range(3):
                sparse_adjacency_matrix[face[i], face[(i + 1) % 3]] = 1
                sparse_adjacency_matrix[face[(i + 1) % 3], face[i]] = 1
                
        return sparse_adjacency_matrix

    def to_pyg_data(self):
        # Convert to PyG Data object
        edge_index = torch.tensor(self.sparse_adjacency_matrix.nonzero(), dtype=torch.long)
        x = torch.tensor(self.mesh.vertices, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index)

    def process_one_file(self, file_path):
        self.load_mesh(file_path)
        self.compute_sparse_adjacency_matrix()
        return self.to_pyg_data()
