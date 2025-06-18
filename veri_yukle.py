from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class DiffStegoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.label_map = {'cover': 0, 'lsb': 1, 's_uniward': 2, 'hugo': 3}

        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            for fname in os.listdir(label_path):
                self.samples.append((os.path.join(label_path, fname), self.label_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label
