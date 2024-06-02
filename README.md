# Semantic Segmentation Project

Bu proje, UDD-VDD veri setini kullanarak iniş yapan hava araçları için bir semantik segmentasyon uygulaması geliştirmeyi amaçlamaktadır. Proje kurulumu ve çalıştırılması için aşağıdaki talimatlar izlenebilir.

## Ön Koşullar

Python 3.8'in yüklü olduğundan emin olun.

## Kurulum

Projeyi çalıştırmadan önce, aşağıdaki komutları çalıştırarak gerekli kütüphaneleri yükleyin:

```bash
# Install PyTorch
!pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0

# Install mim
!pip install -U openmim

# Install mmengine
!mim install mmengine

# Install MMCV
!mim install mmcv==2.0.0rc4

# Install ftfy
!pip install ftfy

# Navigate to the project directory
cd eren_coskun_yzup_baykar

# Install the project in editable mode
!pip install -e .

```

## Linkteki iter_10000.pth weight dosyasını indirin ve configs_and_weights klasörüne kopyalayın

[Link](https://drive.google.com/drive/folders/1TFaTlZe3Wk8BxmbFO1c3SRq_QFhGVkHL?usp=sharing)

## Test Resimleri için Nasıl Çalıştırılır

```bash
python test_run_for_images.py --input_folder path_to_input_images --output_folder path_to_save_segmented_images
```

### Argümanlar

- `--input_folder`: Girdi resimlerinin bulunduğu klasör yolu. default= 'test_videos/test_video_frames'
- `--output_folder`: Segmentlenmiş resimlerin kaydedileceği klasör yolu. default='default_output_folder_video'

`path_to_input_images`, `path_to_save_segmented_images`, `path_to_output_video` sisteminizde dosyalarınızı saklamak istediğiniz gerçek yollarla değiştirin.

## Test Videosu için Nasıl Çalıştırılır
### Custom videonuzu modelimizde test etmeniz için öncelikle test_videos klasörü içindeki extract_video_frame.py scriptini çalıştırarak video frameler haline getirilmelidir. Default video frameleri test_videos klasöründe bulunmaktadır. 

1. Terminalinizi veya komut istemcinizi açın.
2. Proje dizinine gidin.
3. Aşağıdaki komutu kullanarak betiği çalıştırın:

```bash
python test_run_for_videos.py --input_folder path_to_input_images --output_folder path_to_save_segmented_images --video_output_path path_to_output_video
```

### Argümanlar

- `--input_folder`: Girdi resimlerinin bulunduğu klasör yolu. default= 'test_videos/test_video_frames'
- `--output_folder`: Segmentlenmiş resimlerin kaydedileceği klasör yolu. default='default_output_folder_video'
- `--video_output_path`: Çıkış video dosyası yolu. default='default_output_video_path'
- `--frame_rate`: Çıkış videosunun kare hızı (varsayılan 30).
- `--repeat_frame`: Videoda her karenin tekrarlanma sayısı (varsayılan 10).

`girdi_resimleri_klasör_yolu, segmentlenmiş_resimleri_kaydet_klasör_yolu ve çıkış_video_yolunı sisteminizde dosyalarınızı saklamak istediğiniz gerçek yollarla değiştirin.`
