import cv2
import os

# Video dosyasının yolu
video_path = '15021673374_1080p.mp4'

# Çıktı klasörü
output_folder = 'test_video_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Video dosyasını okuyun
cap = cv2.VideoCapture(video_path)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Frame'i kaydet
    frame_filename = os.path.join(output_folder, f"{frame_count:06d}.png")
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1

# Kaynakları serbest bırakın
cap.release()

print(f"Toplam {frame_count} frame kaydedildi.")
