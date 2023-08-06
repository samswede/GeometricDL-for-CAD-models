import unittest
import trimesh
import torch
from torch_geometric.data import Data
import scipy.sparse as sp
from preprocessing.data_pipeline import DataPipeline  # Make sure to import the DataPipeline class

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.data_folder_path = '/Users/samuelandersson/Dev/CodeSprint/GeometricDL-for-CAD-models/data/raw/ModelNet10' # Provide the path to the data folder
        self.example_file_path = '/Users/samuelandersson/Dev/CodeSprint/GeometricDL-for-CAD-models/data/raw/ModelNet10/chair/train/chair_0001.off'
        self.pipeline = DataPipeline(self.data_folder_path)

    def test_instance_data_converter(self):
        self.assertTrue(self.pipeline is not None)

    def test_load_mesh_from_file(self):
        mesh = self.pipeline.load_mesh_from_file(self.example_file_path)
        self.assertIsInstance(mesh, trimesh.base.Trimesh)
        
    def test_compute_sparse_adjacency_matrix(self):
        mesh = self.pipeline.load_mesh_from_file(self.example_file_path)
        
        # Get vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces

        sparse_adjacency_matrix = self.pipeline.compute_sparse_adjacency_matrix(vertices, faces)

        self.assertIsInstance(sparse_adjacency_matrix, sp.csr_matrix)

    # def test_to_pyg_data(self):
    #     mesh = self.pipeline.load_mesh_from_file(self.example_file_path)
    #     # Get vertices and faces
    #     vertices = mesh.vertices
    #     faces = mesh.faces
    #     sparse_adjacency_matrix = self.pipeline.compute_sparse_adjacency_matrix(vertices, faces)
    #     pyg_data = self.pipeline.to_pyg_data(sparse_adjacency_matrix, vertices)
    #     self.assertIsInstance(pyg_data, Data)

    def assertDataEqual(self, data1, data2):
        self.assertEqual(data1.x.shape, data2.x.shape)
        self.assertEqual(data1.edge_index.shape, data2.edge_index.shape)

    def test_process_coo_vectorised(self):
        pyg_data = self.pipeline.process_one_file_coo(self.example_file_path)
        pyg_data_vectorised = self.pipeline.process_one_file_coo_vectorised(self.example_file_path)

        self.assertDataEqual(pyg_data, pyg_data_vectorised)
