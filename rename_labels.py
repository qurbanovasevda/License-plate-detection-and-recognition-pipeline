import re
from pathlib import Path

img_dir, lbl_dir = Path("data/test/images"), Path("data/test/labels")
EXTS = {".jpg", ".jpeg", ".png"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


# normallaşmış şəkil stem-i -> real şəkil stem-i
img_stems = {norm(p.stem): p.stem for p in img_dir.iterdir() if p.suffix.lower() in EXTS}

renamed, unmatched = 0, []
for lbl in sorted(lbl_dir.glob("*.txt")):
    base = re.sub(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", "", lbl.stem, flags=re.I)
    stem = img_stems.get(norm(base))
    if stem is None:
        unmatched.append(lbl.name)
        continue
    target = lbl.with_name(stem + ".txt")
    if target != lbl:
        lbl.rename(target)
        renamed += 1

print("adı dəyişdirilən label:", renamed)
print("uyğun şəkil tapılmayanlar:", unmatched)