from pathlib import Path

import cv2

from pipeline import read_plate

SRC, OUT = Path("data/unseen"), Path("reports/unseen")
OUT.mkdir(parents=True, exist_ok=True)

for p in sorted(SRC.iterdir()):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    img = cv2.imread(str(p))
    plates = read_plate(img)
    print(f"\n{p.name}: {len(plates)} lövhə")
    for pl in plates:
        x1, y1, x2, y2 = pl["box"]
        print(f"   box={pl['box']} conf={pl['conf']} raw='{pl['raw']}' -> '{pl['text']}'")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, pl["text"], (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(OUT / p.name), img)