import os
import shutil

input_dir = 'data/BOSSBase/diff'
output_base = 'dataset'

# Hedeflenen steganografi algoritmalarını tanımla
algorithms = ['lsb', 's_uniward']

for file in os.listdir(input_dir):
    if file.endswith('.png'):
        # Dosya adında algoritma adı geçenleri tespit et
        for algo in algorithms:
            if algo in file.lower():  # Küçük harf dönüşümü olası büyük harf sorunlarını önler
                label_dir = os.path.join(output_base, algo)
                os.makedirs(label_dir, exist_ok=True)
                shutil.copy(os.path.join(input_dir, file), os.path.join(label_dir, file))
                break  # Aynı dosya birden fazla klasöre kopyalanmasın diye döngüden çık
