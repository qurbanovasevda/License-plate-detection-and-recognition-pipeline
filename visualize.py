from pathlib import Path

import cv2

from evaluate import EXTS, IMG_DIR, LBL_DIR, key, load_gt
from pipeline import read_plate

VIS, CROPS = Path("reports/vis"), Path("reports/crops")
VIS.mkdir(parents=True, exist_ok=True)
CROPS.mkdir(parents=True, exist_ok=True)
labels = {key(p.name): p for p in LBL_DIR.glob("*.txt")}

for img_path in sorted(IMG_DIR.iterdir()):
    if img_path.suffix.lower() not in EXTS:
        continue
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    vis = img.copy()
    for (x1, y1, x2, y2) in load_gt(labels[key(img_path.name)], w, h):
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
    for i, p in enumerate(read_plate(img)):
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f'{p["conf"]} {p["text"]}', (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(str(CROPS / f"{img_path.stem}_{i}.jpg"), img[y1:y2, x1:x2])
    cv2.imwrite(str(VIS / f"{img_path.stem}.jpg"), vis)
print("hazırdır: reports/vis və reports/crops")