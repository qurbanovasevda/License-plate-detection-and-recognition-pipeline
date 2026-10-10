from pathlib import Path
import csv

img_dir, lbl_dir = Path("data/test/images"), Path("data/test/labels")
imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
lbls = {p.stem for p in lbl_dir.glob("*.txt")}

print("şəkil:", len(imgs), "label:", len(lbls))
print("labeli olmayan şəkillər:", sorted(set(imgs) - lbls))
print("şəkli olmayan labellər:", sorted(lbls - set(imgs)))

csv_path = Path("data/test/plates.csv")
if not csv_path.exists():
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "plate_text"])
        for stem in sorted(imgs):
            w.writerow([imgs[stem].name, ""])
    print("plates.csv şablonu yaradıldı")


