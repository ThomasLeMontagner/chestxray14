from src.train import fit
from src.data_module import get_train_val_dataloaders
from torchvision.models import densenet121
import torch.nn as nn
import torch.optim as optim
import torch

# Get data
train_loader, val_loader, label_names = get_train_val_dataloaders(
    csv_path="../data/Data_Entry_2017_v2020.csv",
    image_dir="../data/images",
    batch_size=32
)

# Define model
model = densenet121(pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, len(label_names)),
    nn.Sigmoid()  # sigmoid for multi-label
)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.BCELoss()  # multi-label binary loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train
fit(model, train_loader, val_loader, optimizer, criterion, device, epochs=10)
