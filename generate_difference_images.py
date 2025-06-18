import os
import numpy as np
from PIL import Image

# Klasör yolları
cover_dir = 'data/BOSSBase/cover'
stego_dir = 'data/BOSSBase/stego'
diff_dir = 'data/BOSSBase/diff'

# Görsel farkları büyütmek için katsayı
amplify_factor = 15  # 10–30 arasında değerler denenebilir

os.makedirs(diff_dir, exist_ok=True)

for filename in os.listdir(stego_dir):
    if filename.endswith('.png'):
        base_name = filename.split('_')[0]
        cover_path = os.path.join(cover_dir, f"{base_name}.pgm")
        stego_path = os.path.join(stego_dir, filename)
        diff_path = os.path.join(diff_dir, filename)

        try:
            cover_img = Image.open(cover_path).convert('L')
            stego_img = Image.open(stego_path).convert('L')

            cover_np = np.array(cover_img, dtype=np.int16)
            stego_np = np.array(stego_img, dtype=np.int16)

            diff_np = np.abs(stego_np - cover_np)

            # Görselleştirme için farkı büyüt
            amplified_diff = np.clip(diff_np * amplify_factor, 0, 255).astype(np.uint8)

            diff_img = Image.fromarray(amplified_diff)
            diff_img.save(diff_path)

            print(f"Fark görüntüsü oluşturuldu: {diff_path}")
        except Exception as e:
            print(f"Hata oluştu: {filename} -> {e}")


if __name__ == "__main__":
    main()