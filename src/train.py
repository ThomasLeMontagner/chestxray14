from torchvision import transforms
from src.dataset import ChestXray14Dataset

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

dataset = ChestXray14Dataset(
    csv_path="../data/Data_Entry_2017.csv",
    image_dir="../data/images",
    transform=transform,
)
