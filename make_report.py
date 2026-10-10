"""README-dəki end-to-end cədvəlini nəticə CSV-sindən avtomatik doldurur.

İstifadə:
    python make_report.py                      # default CSV
    python make_report.py reports/baska.csv    # başqa CSV
"""
import csv
import sys
from pathlib import Path

CSV_PATH = Path(sys.argv[1] if len(sys.argv) > 1
                else "reports/results_conf0.25_sizes640-1280.csv")
README = Path("README.md")
START, END = "<!-- TABLE_START -->", "<!-- TABLE_END -->"

STATUS = {
    "ok": "✅ düzgün",
    "ocr_error": "❌ OCR səhvi",
    "detector_miss": "❌ detektor qaçırdı",
    "ocr_skipped": "— (oxunmur, hesaba daxil deyil)",
}


def cell(v):
    return str(v).replace("|", "/") if v not in ("", None) else "—"


rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig", newline="")))
lines = [
    "| Şəkil | GT box | Aşkarlanmış box | OCR (xam) | OCR (təmiz) | Real mətn | IoU | Nəticə |",
    "|---|---|---|---|---|---|---|---|",
]
for r in rows:
    lines.append("| " + " | ".join([
        f"`{r['image']}`", r["n_gt"], cell(r["pred_box"]), cell(r["ocr_raw"]),
        cell(r["ocr_text"]), cell(r["gt_text"]), cell(r["main_iou"]),
        STATUS.get(r["status"], r["status"]),
    ]) + " |")

text = README.read_text(encoding="utf-8")
if START not in text or END not in text:
    raise SystemExit("README-də TABLE_START / TABLE_END işarələri tapılmadı")
head, rest = text.split(START, 1)
_, tail = rest.split(END, 1)
README.write_text(head + START + "\n" + "\n".join(lines) + "\n" + END + tail,
                  encoding="utf-8")
print(f"{len(rows)} sətir README-yə yazıldı ({CSV_PATH})")
