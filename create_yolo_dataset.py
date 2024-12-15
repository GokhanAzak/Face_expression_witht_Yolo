import os
import cv2
import numpy as np
import pandas as pd

# Proje klasör yolu (ihtiyaç halinde güncelleyin)
DATA_DIR = "data"  # fer2013.csv dosyasının bulunduğu klasör
CSV_PATH = os.path.join(DATA_DIR, "fer2013.csv")

IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# TXT listelerini kaydetmek için
TRAIN_LIST_PATH = os.path.join(DATA_DIR, "train.txt")
VAL_LIST_PATH = os.path.join(DATA_DIR, "val.txt")
TEST_LIST_PATH = os.path.join(DATA_DIR, "test.txt")


def main():
    df = pd.read_csv(CSV_PATH)

    train_files = []
    val_files = []
    test_files = []

    for idx, row in df.iterrows():
        emotion = int(row['emotion'])  # 0..6
        pixels = row['pixels']  # "70 80 90 ..."
        usage = row['Usage']  # "Training" / "PublicTest" / "PrivateTest"

        # Piksel verisini array haline getirme (48x48)
        pixel_array = np.array(pixels.split(), dtype='uint8').reshape(48, 48)

        # 3 kanallı format (YOLO genelde RGB/BGR kullanıyor)
        image_bgr = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)

        # .jpg dosyası adı ve yolu
        img_filename = f"{usage}_{idx}.jpg"
        img_path = os.path.join(IMAGES_DIR, img_filename)

        # Görseli diske kaydet
        cv2.imwrite(img_path, image_bgr)

        # YOLO formatı: class x_center y_center width height (normalize edilmiş)
        # FER2013'de yüz tüm resmi kapladığı için bounding box [0,0,1,1] gibi oluyor
        class_id = emotion
        x_center, y_center, w, h = 0.5, 0.5, 1.0, 1.0

        # .txt etiketi kaydet
        label_filename = f"{usage}_{idx}.txt"
        label_path = os.path.join(LABELS_DIR, label_filename)
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

        # Listelere ekle
        if usage == 'Training':
            train_files.append(img_path)
        elif usage == 'PublicTest':
            val_files.append(img_path)
        elif usage == 'PrivateTest':
            test_files.append(img_path)

    # Artık train.txt, val.txt, test.txt dosyaları oluşturabiliriz
    with open(TRAIN_LIST_PATH, 'w') as f:
        for path in train_files:
            f.write(path + "\n")

    with open(VAL_LIST_PATH, 'w') as f:
        for path in val_files:
            f.write(path + "\n")

    with open(TEST_LIST_PATH, 'w') as f:
        for path in test_files:
            f.write(path + "\n")

    print("YOLO formatına dönüştürme tamamlandı!")


if __name__ == "__main__":
    main()
