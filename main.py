import os
from tqdm import tqdm
from utils.bossbase_loader import load_cover_images
from utils.embedding_simulator import embed_and_save

ALGORITHMS = ['wow']  # hugo, suniward eklenecek
BPPS = [0.2, 0.4]


def main():
    cover_folder = 'data/BOSSBase/cover'
    cover_images = load_cover_images(cover_folder)

    for alg in ALGORITHMS:
        for bpp in BPPS:
            output_dir = f'data/BOSSBase/stego/{alg}_{bpp}'
            for img_path in tqdm(cover_images, desc=f"{alg.upper()} - {bpp} bpp"):
                embed_and_save(alg, img_path, output_dir, bpp)


if __name__ == "__main__":
    main()
