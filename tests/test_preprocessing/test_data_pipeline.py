import unittest
import trimesh
import torch
from torch_geometric.data import Data
import scipy.sparse as sp
from preprocessing.data_pipeline import DataPipeline  # Make sure to import the DataPipeline class

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.data_folder_path = '/Users/samuelandersson/Dev/CodeSprint/GeometricDL-for-CAD-models/data/raw/ModelNet10' # Provide the path to the data folder
        self.pipeline = DataPipeline(self.data_folder_path)

    def test_instance_data_converter(self):
        self.assertTrue(self.pipeline is not None)

    def test_load_mesh_from_file(self):
        file_path = 'path_to_example_file.off' # Provide the path to an example OFF file
        mesh = self.pipeline.load_mesh_from_file(file_path)
        self.assertIsInstance(mesh, trimesh.base.Trimesh)
        
    # def test_compute_adjacency_matrix(self):
    #     file_path = 'path_to_example_file.off' # Provide the path to an example OFF file
    #     self.pipeline.mesh = self.pipeline.load_mesh_from_file(file_path)
    #     self.pipeline.compute_adjacency_matrix()
    #     self.assertIsInstance(self.pipeline.adjacency_matrix, sp.csr.csr_matrix)

    # def test_to_pyg_data(self):
    #     file_path = 'path_to_example_file.off' # Provide the path to an example OFF file
    #     self.pipeline.mesh = self.pipeline.load_mesh_from_file(file_path)
    #     self.pipeline.compute_adjacency_matrix()
    #     pyg_data = self.pipeline.to_pyg_data()
    #     self.assertIsInstance(pyg_data, Data)

    # def test_process_one_file(self):
    #     file_path = 'path_to_example_file.off' # Provide the path to an example OFF file
    #     pyg_data = self.pipeline.process_one_file(file_path)
    #     self.assertIsInstance(pyg_data, Data)
