
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os

# Tekil veri artırma işlemleri
transforms = [
    # A.RandomRain(p=1.0),
    # A.RandomSnow(p=0.2),
    A.RandomShadow(p=1.0),
    # A.RandomFog(p=1.0, fog_coef_lower=0.2, fog_coef_upper=0.4, alpha_coef=0.1,)
    # A.Blur(blur_limit=3, p=1.0),
    # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    # A.RandomGamma(gamma_limit=(80, 120), p=1.0),
    # A.VerticalFlip(p=1.0),
    # A.HorizontalFlip(p=1.0),
    # A.RandomRotate90(p=1.0)
]

# Görüntü ve maske dosyalarının bulunduğu klasörler
image_folder = '../data/UDD-VDD/train/src'
mask_folder = '../data/UDD-VDD/train/gt'

# Dönüştürülmüş görüntü ve maske dosyalarının kaydedileceği klasörler
augmented_image_folder = 'data_augmentation_last_clache2/train/src'
augmented_mask_folder = 'data_augmentation_last_clache2/train/gt'

# Klasörlerin var olup olmadığını kontrol et ve yoksa oluştur
os.makedirs(augmented_image_folder, exist_ok=True)
os.makedirs(augmented_mask_folder, exist_ok=True)

# Görüntü ve maske dosyalarını yükle ve veri artırma işlemlerini uygula
image_files = sorted(os.listdir(image_folder))
mask_files = sorted(os.listdir(mask_folder))

count = 1
for img_file, mask_file in zip(image_files, mask_files):
    img_path = os.path.join(image_folder, img_file)
    mask_path = os.path.join(mask_folder, mask_file)

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Orijinal görüntü ve maskeyi kaydet
    cv2.imwrite(os.path.join(augmented_image_folder, f'{count:06d}.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(augmented_mask_folder, f'{count:06d}.png'), mask)
    count += 1

    # Her bir dönüşümü ayrı ayrı uygula ve dönüştürülmüş görüntü ve maskeyi kaydet
    for transform in transforms:
        transformed = transform(image=image, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]

        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(augmented_image_folder, f'{count:06d}.png'), transformed_image)
        cv2.imwrite(os.path.join(augmented_mask_folder, f'{count:06d}.png'), transformed_mask)
        count += 1
