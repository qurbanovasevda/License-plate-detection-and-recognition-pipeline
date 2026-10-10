import cv2

from evaluate import (EXTS, IMG_DIR, IOU_THR, LBL_DIR, iou, key, levenshtein,
                      load_gt, load_gt_text)
from pipeline import (ALLOW, clean_plate, crop_plates, detector, fix_by_position,
                      reader, strip_country_tag)


def v_current(c):
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def v_color(c):
    return c


def v_gray3(c):
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


def v_h100(c):
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    s = 100 / g.shape[0]
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(g, 20, 20, 20, 20, cv2.BORDER_REPLICATE)


variants = {
    "cari (boz+2x+CLAHE)": v_current,
    "xam rəngli crop": v_color,
    "boz 3x (CLAHE-siz)": v_gray3,
    "hündürlük 100px + kənar": v_h100,
}

gt_text = load_gt_text()
labels = {key(p.name): p for p in LBL_DIR.glob("*.txt")}
stats = {n: {"ok": 0, "cer": 0.0} for n in variants}
total = 0

for img_path in sorted(IMG_DIR.iterdir()):
    if img_path.suffix.lower() not in EXTS:
        continue
    k = key(img_path.name)
    truth = gt_text.get(k, "")
    if truth in ("", "UNREADABLE"):
        continue
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    main = max(load_gt(labels[k], w, h), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    res = detector.predict(img, conf=0.25, verbose=False)[0]
    boxes = [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in res.boxes]
    cands = [c for c in crop_plates(img, boxes) if iou(main, c[0]) >= IOU_THR]
    if not cands:
        continue
    crop = max(cands, key=lambda c: iou(main, c[0]))[2]
    total += 1
    for name, fn in variants.items():
        texts = reader.readtext(fn(crop), detail=0, allowlist=ALLOW)
        pred = fix_by_position(strip_country_tag(clean_plate(" ".join(texts))))
        stats[name]["ok"] += pred == truth
        stats[name]["cer"] += levenshtein(pred, truth) / len(truth)

print(f"\nDetektorun tapdığı və oxunan lövhə sayı: {total}")
for name, s in stats.items():
    print(f"{name:<28} exact match {s['ok']}/{total}   mean CER {s['cer'] / total:.3f}")