from utils.performance import Timer
from preprocessing.data_pipeline import DataPipeline

with Timer('initialise'):
    data_root_path = '/Users/samuelandersson/Dev/CodeSprint/GeometricDL-for-CAD-models/data/raw/ModelNet10'
    save_path = '/Users/samuelandersson/Dev/CodeSprint/GeometricDL-for-CAD-models/data/processed'

    file_path = '/Users/samuelandersson/Dev/CodeSprint/GeometricDL-for-CAD-models/data/raw/ModelNet10/chair/train/chair_0001.off'

    pipeline = DataPipeline(data_root_path, save_path)


# LOAD ALL DATA INTO PYTORCH DATA
with Timer('PROCESS ALL DATA INTO PYTORCH DATA'):
    pytorch_dataset = pipeline.process_dataset()

# Split train and test set
with Timer('Split train and test set'):
    train_data, test_data = pipeline.split_dataset(pytorch_dataset)

# LOAD INTO PYTORCH Geometric DATALOADER
with Timer('LOAD INTO PYTORCH Geometric DATALOADER'):
    train_loader, test_loader = pipeline.create_dataloaders(train_data, test_data)

# SAVE PYTORCH DATALOADER AS .pt FILE
with Timer('SAVE PYTORCH DATALOADER AS .pt FILE'):
    pipeline.save_dataset(train_data, 'train_loader.pt')
    pipeline.save_dataset(test_data, 'test_loader.pt')

# LOAD PYTORCH DATASET INTO DATALOADER FROM FILES
with Timer('LOAD PYTORCH DATASET INTO DATALOADER FROM FILES'):
    train_loader_loaded = pipeline.load_dataloader('train_loader.pt')
    test_loader_loaded = pipeline.load_dataloader('test_loader.pt')

    