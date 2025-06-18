import torch
from a.model.model import Srnet  # SRNet sınıfını ayrı bir dosyadaysa import et
from PIL import Image
import torchvision.transforms as transforms

# 1. Modeli yükle
model = Srnet()
model.load_state_dict(torch.load('srnet_weights.pth', map_location=torch.device('cpu')))
model.eval()

# 2. Parametreleri dondur
for param in model.parameters():
    param.requires_grad = False

# 3. Görüntü hazırlama
transform = transforms.Compose([
    transforms.Grayscale(),  # Giriş siyah beyazsa
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

image_path = '../data/test_images/12_s_uniward_0.2.png'
image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # [1, 1, 256, 256]

# 4. Sınıflandır
output = model(image)
_, predicted = torch.max(output.data, 1)

label_map = {0: 'cover', 1: 'lsb', 2: 's_uniward'}
print("Bu fark görüntüsü şuna ait:", label_map[predicted.item()])
