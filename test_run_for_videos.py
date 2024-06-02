import os
import cv2
import argparse
from mmseg.apis import init_model, inference_model, show_result_pyplot

# Sabit yollar
CONFIG_PATH = 'configs_and_weights/pspnet_r50-d8_udd-vdd-512x1024.py'
CHECKPOINT_PATH = 'configs_and_weights/iter_10000_pspnet.pth'

def main(input_folder, output_folder, video_output_path, frame_rate=30, repeat_frame=10):
    # Çıkış klasörü oluştur
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Modeli yükle
    model = init_model(CONFIG_PATH, CHECKPOINT_PATH, device='cpu')

    # Input klasöründeki tüm dosyalar için
    for img_filename in sorted(os.listdir(input_folder)):
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

    # Video oluşturma kısmı
    frame_files = sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if frame_files:
        first_frame = cv2.imread(frame_files[0])
        height, width, layers = first_frame.shape
        video = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            for _ in range(repeat_frame):  # Her frame'i belirlenen sayıda tekrarla
                video.write(frame)
        video.release()
        print("Video oluşturuldu.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic segmentation on video frames")
    parser.add_argument("--input_folder", default='test_videos/test_video_frames', help="Path to the folder containing input images")
    parser.add_argument("--output_folder", default='default_output_folder_video', help="Path to the folder to save output images")
    parser.add_argument("--video_output_path", default='segmented_output_video.mp4', help="Path to the output video file")
    parser.add_argument("--frame_rate", type=int, default=30, help="Frame rate of the output video")
    parser.add_argument("--repeat_frame", type=int, default=10, help="Number of times to repeat each frame in the video")

    args = parser.parse_args()
    
    main(args.input_folder, args.output_folder, args.video_output_path, args.frame_rate, args.repeat_frame)
