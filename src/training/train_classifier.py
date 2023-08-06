import torch
from torch import nn
from torch_geometric.data import DataLoader
from torch.optim import Adam

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

# # Define test function
# def test_classifier(model, loader):
#     model.eval()
#     correct = 0
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             output = model(data)
#             pred = output.argmax(dim=1)
#             correct += int((pred == data.y).sum())
#     return correct / len(loader.dataset)

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


data_root_path = '/Users/samuelandersson/Dev/CodeSprint/GeometricDL-for-CAD-models/data/raw/ModelNet10'
save_path = '/Users/samuelandersson/Dev/CodeSprint/GeometricDL-for-CAD-models/data/processed'

pipeline = DataPipeline(data_root_path, save_path)

# DataLoader
train_loader = pipeline.load_dataloader('train_loader.pt')
test_loader = pipeline.load_dataloader('test_loader.pt')


# Parameters
num_node_features = 6
num_classes = 10
hidden_dim = 64


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SagPoolGAT(num_node_features, num_classes, hidden_dim).to(device)
optimizer = Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()  # Define appropriate loss function


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


