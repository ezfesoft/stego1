import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import conseal as cl  # pip install conseal



# --- Steganografi Fonksiyonları ---

def generate_random_payload(payload_size_bits):
    """
    Belirtilen boyutta rastgele ikili (0 ve 1) bir mesaj oluşturur.
    """
    return np.random.randint(2, size=payload_size_bits)

def embed_suniward(cover_image_path, payload_bits, output_path, seed=None):
    """
    S-UNIWARD algoritmasını kullanarak bir mesajı görüntüye gömer (simülasyon düzeyinde).

    Args:
        cover_image_path (str): Orijinal kapak görüntüsünün yolu.
        payload_bits (np.ndarray): Gömülecek 1 ve 0'lardan oluşan mesaj (bpp'e çevrilerek kullanılır).
        output_path (str): Stego görüntünün kaydedileceği yol.
        seed (int, optional): Rastgelelik için tohum değeri (varsayılan: None).

    Returns:
        bool: Başarılı ise True, hata oluşursa False.
    """
    try:
        # Görüntüyü yükle ve gri tonlamalıya çevir
        image = Image.open(cover_image_path).convert("L")
        image_array = np.array(image, dtype=np.uint8)

        # Kapasiteyi hesapla
        height, width = image_array.shape
        capacity = height * width

        # Mesajın boyutunu kontrol et
        if len(payload_bits) > capacity:
            raise ValueError("Mesaj boyutu görüntünün kapasitesinden büyük!")

        # bpp (bit per pixel) oranını belirle
        bpp = len(payload_bits) / capacity

        # S-UNIWARD gömme (simülasyon düzeyinde)
        stego_array = cl.suniward.simulate_single_channel(x0=image_array, alpha=bpp, seed=seed)

        # Stego görüntüyü kaydet
        stego_image = Image.fromarray(stego_array.astype(np.uint8))
        stego_image.save(output_path)

        return True

    except Exception as e:
        print(f"Hata: {cover_image_path} işlenirken sorun oluştu: {e}")
        return False

def embed_lsb(cover_image_path, payload_bits, output_path):
    """
    LSB (Least Significant Bit) yöntemini kullanarak bir mesaji görüntüye gömer.

    Args:
        cover_image_path (str): Orijinal kapak görüntüsünün yolu.
        payload_bits (np.array): Gömülecek 1 ve 0'lardan oluşan mesaj.
        output_path (str): Stego görüntünün kaydedileceği yol.
    """
    try:
        # Görüntüyü aç ve NumPy dizisine dönüştür
        image = Image.open(cover_image_path).convert("L")  # Gri tonlamalı olduğundan emin ol
        image_array = np.array(image, dtype=np.uint8)

        # Gömme kapasitesini kontrol et
        height, width = image_array.shape
        capacity = height * width
        if len(payload_bits) > capacity:
            raise ValueError("Mesaj boyutu görüntünün kapasitesinden büyük!")

        # Görüntü dizisini tek boyutlu hale getirerek işlemi kolaylaştır
        flat_image_array = image_array.flatten()

        # Her bir mesaj bitini, pikselin en anlamsız bitine (LSB) göm
        for i in range(len(payload_bits)):
            pixel_value = flat_image_array[i]
            message_bit = payload_bits[i]

            # Pikselin son bitini temizle (...11111110) ve mesaj bitini ekle
            flat_image_array[i] = (pixel_value & 0xFE) | message_bit

        # Tek boyutlu diziyi orijinal görüntü boyutlarına geri getir
        stego_image_array = flat_image_array.reshape((height, width))

        # Yeni (stego) görüntüyü oluştur ve kaydet
        # PNG gibi kayıpsız (lossless) bir format kullanmak LSB için kritiktir.
        stego_image = Image.fromarray(stego_image_array)
        stego_image.save(output_path)

        return True

    except Exception as e:
        print(f"Hata: {cover_image_path} işlenirken sorun oluştu: {e}")
        return False


# --- Ana İşlem Betiği (Batch Script) ---

def main():
    """
    Ana betik: BOSSbase görüntüleri üzerinde döngüye girer,
    belirtilen algoritma ve gömme oranları ile stego görüntüler üretir.
    """
    # --- Parametreler ---
    COVER_DIR = "data/BOSSBase/cover/"
    STEGO_DIR = "data/BOSSBase/stego/"

    # Python'da bulduğunuz kütüphanelere göre bu listeyi güncelleyebilirsiniz.
    # Şimdilik sadece LSB örneğini kullanıyoruz.
    #ALGORITHMS = ["lsb"]
    ALGORITHMS = ["s_uniward"]

    # Gömme oranları (bits per pixel)
    PAYLOADS_BPP = [0.2, 0.4]

    # --- Klasör Kontrolü ---
    if not os.path.exists(STEGO_DIR):
        print(f"'{STEGO_DIR}' klasörü oluşturuluyor...")
        os.makedirs(STEGO_DIR)

    # --- İşlem Başlangıcı ---
    cover_image_paths = glob.glob(os.path.join(COVER_DIR, "*.pgm"))

    if not cover_image_paths:
        print(f"Hata: '{COVER_DIR}' klasöründe .pgm uzantılı kapak görüntüsü bulunamadı.")
        return

    print(f"Toplam {len(cover_image_paths)} kapak görüntüsü bulundu.")
    print("Stego görüntü üretimi başlıyor...")

    # tqdm ile ilerleme çubuğu oluştur
    for cover_path in tqdm(cover_image_paths, desc="Kapak Görüntüleri"):
        try:
            # Görüntü boyutlarını al (payload boyutunu hesaplamak için)
            with Image.open(cover_path) as img:
                width, height = img.size

            base_name = os.path.splitext(os.path.basename(cover_path))[0]

            for algo in ALGORITHMS:
                for payload_bpp in PAYLOADS_BPP:
                    # Çıktı dosyasının adını ve yolunu belirle
                    # Örn: 1001_lsb_0.4.png
                    output_filename = f"{base_name}_{algo}_{payload_bpp}.png"
                    output_path = os.path.join(STEGO_DIR, output_filename)

                    # Eğer stego görüntü zaten varsa bu adımı atla
                    if os.path.exists(output_path):
                        continue

                    # 1. Gömülecek mesaj boyutunu hesapla
                    payload_size_in_bits = int(width * height * payload_bpp)

                    # 2. Rastgele mesajı oluştur
                    payload = generate_random_payload(payload_size_in_bits)

                    # 3. Steganografi fonksiyonunu çağır
                    if algo == "lsb":
                        embed_lsb(cover_path, payload, output_path)
                    elif algo == "s_uniward":
                        embed_suniward(cover_path, payload, output_path)


        except Exception as e:
            print(f"\nİşlem sırasında beklenmedik bir hata oluştu: {cover_path}, Hata: {e}")

    print("\nStego görüntü üretimi tamamlandı.")
    print(f"Oluşturulan görüntüler '{STEGO_DIR}' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()