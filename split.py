import shutil
from pathlib import Path

VAL_FRACTION = 0.2

root = Path(__file__).parent
img_dir = root / "train" / "images"
lbl_dir = root / "train" / "labels"

# sort by filename so sequential frames stay in order
images = sorted(
    p for p in img_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    and (lbl_dir / (p.stem + ".txt")).exists()
)

n_total = len(images)
n_val = int(n_total * VAL_FRACTION)

# take val as ONE contiguous block from the middle of the sequence,
# so train still covers the start and end of the flight footage
start = (n_total - n_val) // 2
val_images = images[start : start + n_val]

(root / "valid" / "images").mkdir(parents=True, exist_ok=True)
(root / "valid" / "labels").mkdir(parents=True, exist_ok=True)

for img in val_images:
    lbl = lbl_dir / (img.stem + ".txt")
    shutil.move(str(img), root / "valid" / "images" / img.name)
    shutil.move(str(lbl), root / "valid" / "labels" / lbl.name)

print(f"Total with labels: {n_total}")
print(f"Moved to valid:    {len(val_images)}  (frames {start}..{start + n_val - 1})")
print(f"Remaining train:   {n_total - len(val_images)}")