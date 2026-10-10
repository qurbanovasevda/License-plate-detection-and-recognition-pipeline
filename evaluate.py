import os
import csv
import re
from pathlib import Path

import cv2

from pipeline import read_plate

IMG_DIR = Path("data/test/images")
LBL_DIR = Path("data/test/labels")
CSV_PATH = Path("data/test/plates.csv")
OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)
IOU_THR = 0.5
CONF = float(os.environ.get("CONF", 0.25))
SIZES = (640, 1280)
EXTS = {".jpg", ".jpeg", ".png"}


def key(filename):
    """images9.jgp.jpg və images9-jgp.jpg eyni açara düşsün."""
    return re.sub(r"[^a-z0-9]", "", Path(filename).stem.lower())


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def load_gt(path, w, h):
    boxes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, bw, bh = map(float, line.split()[:5])
        boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_gt_text():
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        delim = ";" if ";" in f.readline() else ","
        f.seek(0)
        out = {}
        for row in csv.DictReader(f, delimiter=delim):
            text = row["plate_text"].strip().upper().replace("-", "").replace(" ", "")
            out[key(row["image"])] = text
    return out


def match(gt_boxes, preds):
    """GT ↔ pred bir-bir uyğunlaşdırma (ən yüksək IoU-dan başlayaraq)."""
    pairs = sorted(((iou(g, p["box"]), gi, pi)
                    for gi, g in enumerate(gt_boxes)
                    for pi, p in enumerate(preds)), reverse=True)
    used_g, used_p, result = set(), set(), {}
    for score, gi, pi in pairs:
        if score <= 0 or gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        result[gi] = (pi, score)
    return result


def main():
    gt_text = load_gt_text()
    labels = {key(p.name): p for p in LBL_DIR.glob("*.txt")}

    # Label faylları çatışmayan şəkillərin yoxlanması
    missing = [
        p.name for p in IMG_DIR.iterdir()
        if p.suffix.lower() in EXTS and key(p.name) not in labels
    ]
    if missing:
        raise SystemExit(f"Label tapılmayan şəkillər: {missing}")

    rows, all_ious = [], []
    tp = fp = n_gt = 0

    for img_path in sorted(IMG_DIR.iterdir()):
        if img_path.suffix.lower() not in EXTS:
            continue
        k = key(img_path.name)
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt_boxes = load_gt(labels[k], w, h) if k in labels else []
        preds = read_plate(img, conf=CONF, sizes=SIZES)

        matches = match(gt_boxes, preds)
        for gi in range(len(gt_boxes)):
            score = matches[gi][1] if gi in matches else 0.0
            all_ious.append(score)
        n_gt += len(gt_boxes)
        hit = {gi for gi, (_, s) in matches.items() if s >= IOU_THR}
        tp += len(hit)
        fp += len(preds) - len(hit)

        # Əsas lövhə = ən böyük GT box
        main_i = max(range(len(gt_boxes)),
                     key=lambda i: (gt_boxes[i][2] - gt_boxes[i][0]) * (gt_boxes[i][3] - gt_boxes[i][1]),
                     default=None)
        main_iou, pred = 0.0, None
        if main_i is not None and main_i in matches and matches[main_i][1] >= IOU_THR:
            pred = preds[matches[main_i][0]]
            main_iou = matches[main_i][1]
        elif main_i is not None and main_i in matches:
            main_iou = matches[main_i][1]

        truth = gt_text.get(k, "")
        pred_text = pred["text"] if pred else ""
        if truth in ("", "UNREADABLE"):
            status, correct, cer = "ocr_skipped", "", ""
        elif pred is None:
            status, correct, cer = "detector_miss", False, 1.0
        else:
            correct = pred_text == truth
            cer = round(levenshtein(pred_text, truth) / len(truth), 3)
            status = "ok" if correct else "ocr_error"

        rows.append({
            "image": img_path.name, "n_gt": len(gt_boxes), "n_pred": len(preds),
            "main_iou": round(main_iou, 3),
            "pred_box": pred["box"] if pred else "",
            "ocr_raw": pred["raw"] if pred else "", "ocr_text": pred_text,
            "gt_text": truth, "correct": correct, "cer": cer, "status": status,
        })

    with open(OUT_DIR / f"results_conf{CONF}_sizes{'-'.join(map(str, SIZES))}.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    for r in rows:
        print(f'{r["image"]:<20} IoU={r["main_iou"]:<6} {r["ocr_text"]:<9} '
              f'GT={r["gt_text"]:<11} {r["status"]}')

    scored = [r for r in rows if r["status"] != "ocr_skipped"]
    detected = [r for r in scored if r["status"] in ("ok", "ocr_error")]
    print("\n=== DETECTION ===")
    print(f"GT box sayı: {n_gt} | mean IoU (bütün GT, qaçırılanlar 0): {sum(all_ious)/len(all_ious):.3f}")
    ok = [x for x in all_ious if x >= IOU_THR]
    print(f"mean IoU (tapılanlar, IoU>={IOU_THR}): {sum(ok)/len(ok):.3f}" if ok else "tapılan yoxdur")
    print(f"Recall@{IOU_THR}: {tp/n_gt:.3f} | Precision@{IOU_THR}: {tp/(tp+fp) if tp+fp else 0:.3f}")
    print("\n=== OCR ===")
    if scored:
        em = sum(r["correct"] is True for r in scored)
        print(f"End-to-end exact match: {em}/{len(scored)} = {em/len(scored):.3f}")
        print(f"End-to-end mean CER: {sum(r['cer'] for r in scored)/len(scored):.3f}")
    if detected:
        em_d = sum(r["correct"] is True for r in detected)
        print(f"Tapılanlar üzrə exact match: {em_d}/{len(detected)} = {em_d/len(detected):.3f}")
        print(f"Tapılanlar üzrə mean CER: {sum(r['cer'] for r in detected)/len(detected):.3f}")
    print("\n=== UĞURSUZLUQLAR ===")
    for r in rows:
        if r["status"] in ("detector_miss", "ocr_error"):
            print(f'{r["image"]}: {r["status"]} | GT={r["gt_text"]} | OCR raw="{r["ocr_raw"]}" -> "{r["ocr_text"]}"')


if __name__ == "__main__":
    main()