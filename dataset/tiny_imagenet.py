"""Tiny ImageNet loaders from Lab 01 Exercise 02 / Lab 02.

Validation data works with either the original flat val/images layout
or the class-folder arrangement from the Lab 02 notebook.
"""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class TinyImageNetValidation(Dataset):
    """Read labeled val images without copying or changing the dataset."""
    def __init__(self, root, class_to_idx, transform=None):
        root = Path(root)
        self.transform = transform
        self.classes = sorted(class_to_idx, key=class_to_idx.get)
        self.samples = []
        annotations = root / 'val_annotations.txt'
        image_dir = root / 'images'
        if annotations.exists() and image_dir.is_dir():
            for line in annotations.read_text(encoding='utf-8').splitlines():
                filename, cls, *_ = line.split('	')
                if cls in class_to_idx:
                    img_path = image_dir / filename
                    if not img_path.is_file():
                        raise FileNotFoundError(img_path)
                    self.samples.append((img_path, class_to_idx[cls]))
        else:
            # Lab 02 moves images into val/<class-id>/ and removes val/images.
            for cls, idx in class_to_idx.items():
                folder = root / cls
                for image_path in sorted(folder.glob('*')):
                    if image_path.suffix.lower() in {'.jpeg', '.jpg', '.png'}:
                        self.samples.append((image_path, idx))
        if not self.samples:
            raise RuntimeError(f'No labeled Tiny ImageNet validation images found in {root}')
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            return image, label


def make_loaders(data_root='data/tiny-imagenet-200', image_size=224,
                 batch_size=32, num_workers=2):
    data_root = Path(data_root)
    if not (data_root / 'train').is_dir() or not (data_root / 'val').is_dir():
        raise FileNotFoundError(
            f'Expected train/ and val/ inside {data_root}. '
            'Download and extract tiny-imagenet-200.zip first.'
        )
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_data = ImageFolder(data_root / 'train', transform=transform)
    val_data = TinyImageNetValidation(data_root / 'val',
                                      train_data.class_to_idx,
                                      transform=transform)
    loader_options = dict(batch_size=batch_size, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    return train_loader, val_loader
