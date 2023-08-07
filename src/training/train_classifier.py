import torch
from torch import nn
from torch_geometric.data import DataLoader
from torch.optim import Adam

from models.gnn import GNN
from models.edge_pool_graphsage import EdgePoolGraphSAGE
from models.sag_pool_gat import SagPoolGAT
from preprocessing.data_pipeline import DataPipeline


# Define training function
def train_classifier(model, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module):
    model.train()  # Set the model in training mode
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        batch.x = batch.x.float()  # Convert the input features to float32
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(train_loader.dataset)

# Define test function
def test_classifier(model, loader: DataLoader):
    model.eval()  # Set the model in evaluation mode
    correct = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.x = batch.x.float()  # Convert the input features to float32
            output = model(batch)
            pred = output.argmax(dim=1)
            correct += int((pred == batch.y).sum())

    return correct / len(loader.dataset)

# Define the model saving function
def save_model(model, model_file_name, model_save_folder_path):
    model_path = F'{model_save_folder_path}/{model_file_name}'
    torch.save(model.state_dict(), model_path)


data_root_path = r'C:\Users\Josh\Desktop\Dev\github-repos\GeometricDL-for-CAD-models\data\raw\ModelNet10'
save_path = r'C:\Users\Josh\Desktop\Dev\github-repos\GeometricDL-for-CAD-models\data\processed'

pipeline = DataPipeline(data_root_path, save_path)

# DataLoader
train_loader = pipeline.load_dataloader('train_loader.pt')
test_loader = pipeline.load_dataloader('test_loader.pt')


# Parameters
model_name = 'EdgePoolGraphSAGE_test'
num_node_features = 6
num_classes = 10
hidden_dim = 16


# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')


#model = SagPoolGAT(num_node_features, num_classes, hidden_dim).to(device)
model = EdgePoolGraphSAGE(num_node_features, num_classes, hidden_dim).to(device)
#model = GNN(num_node_features, num_classes, hidden_dim).to(device)
optimizer = Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()  # Define appropriate loss function

print('Saving model...')
save_model(model, model_file_name=f'{model_name}.pt', model_save_folder_path=r'C:\Users\Josh\Desktop\Dev\github-repos\GeometricDL-for-CAD-models\data\trained_models')
print('Model Saved')
# Training loop
print('Starting Training...')
for epoch in range(1, 3):
    print(f'epoch: {epoch}')
    train_loss = train_classifier(model, train_loader, optimizer, criterion)
    print(f'train_loss computed')
    train_acc = test_classifier(model, train_loader)
    print(f'train_acc computed')
    test_acc = test_classifier(model, test_loader)
    print(f'test_acc computed')

    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# i= 1
# for batch in train_loader:
#     print(f'batch: {batch}')
#     print(f'batch.y: {batch.y}')
#     i +=1
#     if i==3:
#         break


