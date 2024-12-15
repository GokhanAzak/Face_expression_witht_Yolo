import cv2
from ultralytics import YOLO

model_path = "runs/detect/yolo_fer2013/weights/best.pt"
model = YOLO(model_path)

# Örnek bir test resmi
img_path = "data/fer2013/images/PrivateTest_102.jpg"
results = model.predict(source=img_path, conf=0.25)

for r in results:
    # r.boxes: tespit edilen bounding box'lar
    # r.probs: her sınıf için olasılıklar
    # r.names: classes isim listesi
    print(r.boxes)
    print(r.probs)
    # YOLOv8 ile gelen bounding box, sınıf, skor vb. bilgilere ulaşabilirsiniz

    # Annotated görüntü oluşturmak için
    annotated_frame = r.plot()  # Yolo otomatik çizim fonksiyonu
    cv2.imshow("YOLO Result", annotated_frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()
