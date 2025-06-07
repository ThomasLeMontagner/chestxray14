import pandas as pd
from PIL import Image
import torch

from src.dataset import ChestXray14Dataset

def test_multi_hot_labels(tmp_path):
    # Create sample dataframe
    df = pd.DataFrame({
        'Image Index': ['img1.png', 'img2.png'],
        'Finding Labels': ['A|B', 'B|C'],
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    # Create dummy images
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ['img1.png', 'img2.png']:
        Image.new('RGB', (10, 10)).save(image_dir / name)

    # Instantiate dataset
    dataset = ChestXray14Dataset(csv_path=str(csv_path), image_dir=str(image_dir))

    # Label names should be inferred and sorted
    assert dataset.label_names == ["A", "B", "C"]

    # Expected labels
    expected = [torch.tensor([1, 1, 0], dtype=torch.float32),
                torch.tensor([0, 1, 1], dtype=torch.float32)]

    # Check each sample
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        assert torch.equal(label, expected[idx])
