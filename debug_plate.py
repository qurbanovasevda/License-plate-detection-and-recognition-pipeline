import sys

import cv2

from pipeline import ALLOW, crop_plates, detector, preprocess, reader

path = sys.argv[1]
img = cv2.imread(path)
print("şəkil ölçüsü:", img.shape[1], "x", img.shape[0])

# 1) Detektor: çox aşağı conf ilə nə görür?
for imgsz in (640, 1280):
    res = detector.predict(img, conf=0.01, imgsz=imgsz, verbose=False)[0]
    print(f"\nimgsz={imgsz}: {len(res.boxes)} box (conf>=0.01)")
    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        print(f"   conf={float(b.conf[0]):.3f}  box={(x1, y1, x2, y2)}")

# 2) OCR: normal conf ilə tapılan lövhələrdə hər seqment və etibarlılığı
res = detector.predict(img, conf=0.25, verbose=False)[0]
boxes = [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in res.boxes]
for i, (box, c, crop) in enumerate(crop_plates(img, boxes)):
    pre = preprocess(crop)
    cv2.imwrite(f"reports/debug_{i}_crop.png", crop)
    cv2.imwrite(f"reports/debug_{i}_pre.png", pre)
    print(f"\nlövhə {i}: box={box}, crop={crop.shape[1]}x{crop.shape[0]}")
    for _, text, conf in reader.readtext(pre, allowlist=ALLOW):
        print(f"   '{text}'  etibarlılıq={conf:.2f}")