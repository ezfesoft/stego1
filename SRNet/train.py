import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.stegano_dataset import StegoDataset
from a.model.model import Srnet  # github’dan gelen model dosyası
from tqdm import tqdm

# Parametreler
batch_size = 8     # 16 yavaş çalışıyorsa düşür
epochs = 20        # ilk test için 20 epoch yeter
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformlar
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # SRNet girişi
    transforms.ToTensor()
])

# Dataset ve DataLoader
train_dataset = StegoDataset(root_dir='../dataset/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = Srnet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Eğitim döngüsü
for epoch in range(epochs):
    model.train()

    print(f"\nEpoch [{epoch+1}/{epochs}]")
    running_loss = 0.0


    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}")

# Ağırlıkları kaydet
torch.save(model.state_dict(), 'srnet_weights.pth')
