import cv2
import numpy as np
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

DELTA = 30
SECRET = 42     
wmlen = 64      
taillebloc = 8      

def chargerimage(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image non trouvée :" ,path)
    img = img.astype(np.float64)
    return img

def generer(length, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=length)

def appdct(img):
    h, w = img.shape
    imagedct = np.zeros_like(img)
    for i in range(0, h - taillebloc + 1, taillebloc):
        for j in range(0, w - taillebloc + 1, taillebloc):
            block = img[i:i+taillebloc, j:j+taillebloc]
            imagedct[i:i+taillebloc, j:j+taillebloc] = dctn(block, norm='ortho')
    return imagedct

def appidct(imagedct):
    h, w = imagedct.shape
    img = np.zeros_like(imagedct)
    for i in range(0, h - taillebloc + 1, taillebloc):
        for j in range(0, w - taillebloc + 1, taillebloc):
            block = imagedct[i:i+taillebloc, j:j+taillebloc]
            img[i:i+taillebloc, j:j+taillebloc] = idctn(block, norm='ortho')
    return img


def coefficients(imagedct, wm_length, key):
    h, w = imagedct.shape
    nb_blocks_h = h // taillebloc
    nb_blocks_w = w // taillebloc
    mf_positions = [(2,3),(3,2),(3,3),(2,4),(4,2),(4,3),(3,4),(4,4),(2,5),(5,2)]
    coords = []
    rng = np.random.default_rng(key)
    blocks = [(bi, bj) for bi in range(nb_blocks_h) for bj in range(nb_blocks_w)]
    chosen_blocks = rng.choice(len(blocks), size=wm_length, replace=False)

    for idx, block_idx in enumerate(chosen_blocks):
        bi, bj = blocks[block_idx]
        pos = mf_positions[idx % len(mf_positions)]
        row = bi * taillebloc + pos[0]
        col = bj * taillebloc + pos[1]
        coords.append((row, col))

    return coords

def embed_watermark(img, watermark):
    imagedct = appdct(img)
    coords = coefficients(imagedct, len(watermark), SECRET)
    for idx, (r, c) in enumerate(coords):
        coeff = imagedct[r, c]
        bit = watermark[idx]
        q = np.round(coeff / DELTA)
        if bit == 0:
            if int(q) % 2 != 0:
                q += 1
        else:
            if int(q) % 2 == 0:
                q += 1
        imagedct[r, c] = q * DELTA

    watermarked_img = appidct(imagedct)
    watermarked_img = np.clip(watermarked_img, 0, 255)
    return watermarked_img

def extract_watermark(img, wmlen):
    imagedct = appdct(img)
    coords = coefficients(imagedct, wmlen, SECRET)

    extracted = []
    for (r, c) in coords:
        coeff = imagedct[r, c]
        q = int(np.round(coeff / DELTA))
        bit = 0 if q % 2 == 0 else 1
        extracted.append(bit)

    return np.array(extracted)

def attack_gaussian_noise(img, sigma=5):
    noise = np.random.normal(0, sigma, img.shape)
    attacked = np.clip(img + noise, 0, 255)
    return attacked

def attack_jpeg(img, quality=70):
    img_uint8 = img.astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', img_uint8, encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return dec.astype(np.float64)

def calculate_psnr(original, watermarked):
    orig_uint8 = np.clip(original, 0, 255).astype(np.uint8)
    wm_uint8 = np.clip(watermarked, 0, 255).astype(np.uint8)
    return psnr(orig_uint8, wm_uint8, data_range=255)

def calculate_ber(original_wm, extracted_wm):
    errors = np.sum(original_wm != extracted_wm)
    return errors / len(original_wm)

def display_results(original, watermarked, attacked_noise, attacked_jpeg):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ['Image originale', 'Image tatouée', 'Après bruit', 'Après JPEG']
    images = [original, watermarked, attacked_noise, attacked_jpeg]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.clip(img, 0, 255).astype(np.uint8), cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('resultats_watermark.png', dpi=150)
    plt.show()
    print("Figure sauvegardée : resultats_watermark.png")

if __name__ == "__main__":
    IMAGE_PATH = "noiretblanc.png"
    print("=== Système de tatouage numérique QIM-DCT ===\n")
    original = chargerimage(IMAGE_PATH)
    print(f"Image chargée : {original.shape}")
    watermark = generer(wmlen, seed=7)
    print(f"Watermark généré ({wmlen} bits) : {watermark}")
    watermarked = embed_watermark(original, watermark)
    print("Watermark inséré.")
    psnr_val = calculate_psnr(original, watermarked)
    print(f"PSNR (original vs tatoué) : {psnr_val:.2f} dB")
    extracted = extract_watermark(watermarked, wmlen)
    ber_no_attack = calculate_ber(watermark, extracted)
    print(f"\n--- Sans attaque ---")
    print(f"BER : {ber_no_attack:.4f} ({int(ber_no_attack*wmlen)}/{wmlen} bits erronés)")
    attacked_noise = attack_gaussian_noise(watermarked, sigma=5)
    ext_noise = extract_watermark(attacked_noise, wmlen)
    ber_noise = calculate_ber(watermark, ext_noise)
    print(f"\n--- Après bruit gaussien (σ=5) ---")
    print(f"BER : {ber_noise:.4f} ({int(ber_noise*wmlen)}/{wmlen} bits erronés)")
    attacked_jpeg_img = attack_jpeg(watermarked, quality=70)
    ext_jpeg = extract_watermark(attacked_jpeg_img, wmlen)
    ber_jpeg = calculate_ber(watermark, ext_jpeg)
    print(f"\n--- Après compression JPEG (qualité=70) ---")
    print(f"BER : {ber_jpeg:.4f} ({int(ber_jpeg*wmlen)}/{wmlen} bits erronés)")
    display_results(original, watermarked, attacked_noise, attacked_jpeg_img)
    print("\n=== Terminé ===")