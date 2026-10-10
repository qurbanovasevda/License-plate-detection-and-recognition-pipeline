import re
from itertools import combinations
import cv2
import easyocr
from ultralytics import YOLO

# Modellər bir dəfə yüklənir
detector = YOLO("weights/best.pt")
reader = easyocr.Reader(["en"], gpu=False)

TO_DIGIT = {"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"}
TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "4": "A"}
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def crop_plates(img, boxes, pad=0.1):
    h, w = img.shape[:2]
    out = []
    for (x1, y1, x2, y2, conf) in boxes:
        px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
        cx1, cy1 = max(0, int(x1) - px), max(0, int(y1) - py)
        cx2, cy2 = min(w, int(x2) + px), min(h, int(y2) + py)
        raw_box = (int(x1), int(y1), int(x2), int(y2))
        out.append((raw_box, conf, img[cy1:cy2, cx1:cx2]))
    return out


def preprocess(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def clean_plate(raw):
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def _fits(ch, i):
    return ch.isdigit() if i in (0, 1, 4, 5, 6) else ch.isalpha()


def _score(s):
    return sum(_fits(c, i) for i, c in enumerate(s))


def strip_country_tag(t):
    if len(t) <= 7:
        return t
    windows = [t[i:i + 7] for i in range(len(t) - 6)]
    scores = [_score(w) for w in windows]
    best_w = [w for w, s in zip(windows, scores) if s == max(scores)][-1]
    if _score(best_w) == 7 or len(t) > 10:
        return best_w
    best, best_score = best_w, _score(best_w)
    for keep in combinations(range(len(t)), 7):
        cand = "".join(t[i] for i in keep)
        if _score(cand) > best_score:
            best, best_score = cand, _score(cand)
    return best

def fix_by_position(t):
    if len(t) != 7:
        return t
    out = []
    for i, ch in enumerate(t):
        if i in (0, 1, 4, 5, 6):
            out.append(TO_DIGIT.get(ch, ch))
        else:
            out.append(TO_LETTER.get(ch, ch))
    return "".join(out)


def detect(img, conf, sizes=(640, 1280)):
    result = None
    for s in sizes:
        result = detector.predict(img, conf=conf, imgsz=s, verbose=False)[0]
        if len(result.boxes):
            break
    return result


def suppress_nested(boxes, thr=0.8):
    kept = []
    for b in sorted(boxes, key=lambda b: b[4], reverse=True):
        x1, y1, x2, y2 = b[:4]
        area = max(1e-6, (x2 - x1) * (y2 - y1))
        nested = False
        for k in kept:
            ix = max(0, min(x2, k[2]) - max(x1, k[0]))
            iy = max(0, min(y2, k[3]) - max(y1, k[1]))
            if ix * iy / area >= thr:
                nested = True
                break
        if not nested:
            kept.append(b)
    return kept


def ocr_text(img):
    results = reader.readtext(img, detail=1, allowlist=ALLOW)
    results.sort(key=lambda r: min(p[0] for p in r[0]))
    return " ".join(r[1] for r in results)

def read_plate(image, conf=0.25, sizes=(640,1280)):
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        return []

    result = detect(img, conf, sizes)
    boxes = [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in result.boxes]
    boxes = suppress_nested(boxes)
    if not boxes:  # 0 aşkarlama: xəta yox, boş siyahı
        return []

    plates = []
    for box, c, crop in crop_plates(img, boxes):
        if crop.size == 0:
            continue
        
        # EasyOCR yalnız təyin olunmuş simvolları (ALLOW) oxuyur
        raw = ocr_text(preprocess(crop))
        
        plates.append({
            "box": box,
            "conf": round(c, 3),
            "raw": raw,
            "text": fix_by_position(strip_country_tag(clean_plate(raw))),
        })
    return plates

def read_plate_text(image, conf=0.25):
    """Şəkil daxil, ən etibarlı lövhənin mətni çıxış (tapılmasa boş sətir)."""
    plates = read_plate(image, conf)
    return max(plates, key=lambda p: p["conf"])["text"] if plates else ""


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(read_plate(sys.argv[1]))