import os
from ultralytics import YOLO


def main():
    # Eğitim parametreleri
    data_yaml = "data.yaml"  # data.yaml dosya yolu
    model_arch = "yolov8n.pt"  # YOLOv8 küçük modelin ön-eğitimli ağırlıkları
    epochs = 30
    batch_size = 16
    img_size = 64  # fer2013 düşük çözünürlüklü, 64 veya 128 kullanılabilir

    # Modeli yükle (ön-eğitimli bir YOLOv8 modeli)
    model = YOLO(model_arch)

    # Eğitim başlat
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        name="yolo_fer2013",  # Çıktı klasörünün adı
    )


if __name__ == "__main__":
    main()
