import os
import argparse
from mmseg.apis import init_model, inference_model, show_result_pyplot

# Sabit yollar
CONFIG_PATH = 'configs_and_weights/pspnet_r50-d8_udd-vdd-512x1024.py'
CHECKPOINT_PATH = 'configs_and_weights/iter_10000_pspnet.pth'

def main(input_folder, output_folder):
    # Çıkış klasörü oluştur
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Modeli yükle
    model = init_model(CONFIG_PATH, CHECKPOINT_PATH, device='cpu')

    # Input klasöründeki tüm dosyalar için
    for img_filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_filename)
        
        # Görüntü dosyasının geçerli bir görüntü olup olmadığını kontrol et
        if not os.path.isfile(img_path) or not img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Model ile tahmin yap
        result = inference_model(model, img_path)
        
        
        # Çıktı dosyasını kaydet
        output_path = os.path.join(output_folder, img_filename)
        show_result_pyplot(model, img_path, result, out_file=output_path, show=False) 
        

    print("Tüm görüntüler işlendi ve kaydedildi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic segmentation on video frames")
    parser.add_argument("--input_folder", default='test_images', help="Path to the folder containing input images")
    parser.add_argument("--output_folder", default='default_output_folder_images', help="Path to the folder to save output images")

    args = parser.parse_args()
    
    main(args.input_folder, args.output_folder)
