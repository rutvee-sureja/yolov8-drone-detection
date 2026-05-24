import json, os, shutil, random, yaml
from collections import defaultdict

random.seed(42)

COCO_IMGS = "/Users/rutveesureja/fiftyone/coco-2017/train/data"
COCO_JSON = "/Users/rutveesureja/fiftyone/coco-2017/train/labels.json"
CUSTOM    = "/Users/rutveesureja/Documents/final_dataset"
OUTPUT    = "/Users/rutveesureja/Documents/final_model_dataset2"

COCO_CLASSES = ["person","car","bus","train","laptop","mouse","keyboard","cell phone","motorcycle"]
COCO_MAP     = {name: i+3 for i, name in enumerate(COCO_CLASSES)}

if os.path.exists(OUTPUT):
    shutil.rmtree(OUTPUT)
for split in ["train", "val", "test"]:
    os.makedirs(f"{OUTPUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT}/labels/{split}", exist_ok=True)

# ── Custom classes (all images) ────────────────────────
all_custom = []
img_dir = f"{CUSTOM}/images"
lbl_dir = f"{CUSTOM}/labels"

for img_file in os.listdir(img_dir):
    if img_file.startswith('.'):
        continue
    stem = os.path.splitext(img_file)[0]
    lbl_path = os.path.join(lbl_dir, stem + ".txt")
    if not os.path.exists(lbl_path):
        parts = stem.rsplit('_', 1)
        if len(parts) == 2:
            num = parts[1].lstrip('0') or '0'
            lbl_path = os.path.join(lbl_dir, parts[0] + "_" + num.zfill(3) + ".txt")
    if not os.path.exists(lbl_path):
        continue
    if stem.startswith("drone"):
        new_id = 0
    elif stem.startswith("bird"):
        new_id = 1
    elif stem.startswith("airplane"):
        new_id = 2
    else:
        continue
    all_custom.append((os.path.join(img_dir, img_file), lbl_path, new_id))

# ── COCO classes (200 each) ────────────────────────────
with open(COCO_JSON) as f:
    coco = json.load(f)

cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
img_id_to_file = {img['id']: img['file_name'] for img in coco['images']}
img_id_to_size = {img['id']: (img['width'], img['height']) for img in coco['images']}

img_anns = defaultdict(list)
for ann in coco['annotations']:
    name = cat_id_to_name.get(ann['category_id'])
    if name in COCO_CLASSES:
        img_anns[ann['image_id']].append(ann)

class_images = defaultdict(list)
for img_id, anns in img_anns.items():
    for ann in anns:
        name = cat_id_to_name[ann['category_id']]
        if name in COCO_CLASSES:
            class_images[name].append(img_id)

all_coco = []
for class_name in COCO_CLASSES:
    imgs = list(set(class_images[class_name]))
    random.shuffle(imgs)
    count = 0
    for img_id in imgs:
        if count >= 200:
            break
        fname = img_id_to_file[img_id]
        src = os.path.join(COCO_IMGS, fname)
        if not os.path.exists(src):
            continue
        W, H = img_id_to_size[img_id]
        anns = img_anns[img_id]
        lines = []
        for ann in anns:
            name = cat_id_to_name[ann['category_id']]
            if name not in COCO_MAP:
                continue
            cls_id = COCO_MAP[name]
            x, y, w, h = ann['bbox']
            cx = (x + w/2) / W
            cy = (y + h/2) / H
            nw = w / W
            nh = h / H
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        if not lines:
            continue
        all_coco.append((src, lines, fname))
        count += 1
    print(f"{class_name}: {count} images")

# ── Combine and split 80/10/10 ─────────────────────────
all_data = []

# Add custom
for img_path, lbl_path, new_id in all_custom:
    all_data.append(("custom", img_path, lbl_path, new_id, None))

# Add coco
for src, lines, fname in all_coco:
    all_data.append(("coco", src, None, None, (lines, fname)))

random.shuffle(all_data)
n = len(all_data)
t = int(n * 0.80)
v = int(n * 0.90)
splits = {"train": all_data[:t], "val": all_data[t:v], "test": all_data[v:]}

for split, items in splits.items():
    for item in items:
        if item[0] == "custom":
            _, img_path, lbl_path, new_id, _ = item
            img_file = os.path.basename(img_path)
            stem = os.path.splitext(img_file)[0]
            shutil.copy(img_path, f"{OUTPUT}/images/{split}/{img_file}")
            with open(lbl_path) as f:
                lines = f.readlines()
            with open(f"{OUTPUT}/labels/{split}/{stem}.txt", "w") as f:
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        parts[0] = str(new_id)
                        f.write(" ".join(parts) + "\n")
        else:
            _, src, _, _, (lines, fname) = item
            stem = os.path.splitext(fname)[0]
            new_img = f"coco_{stem}{os.path.splitext(fname)[1]}"
            new_lbl = f"coco_{stem}.txt"
            shutil.copy(src, f"{OUTPUT}/images/{split}/{new_img}")
            with open(f"{OUTPUT}/labels/{split}/{new_lbl}", "w") as f:
                f.write("\n".join(lines))

print(f"\nTotal: {n}  Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test: {len(splits['test'])}")

# ── data.yaml ──────────────────────────────────────────
names = {0:"drone", 1:"bird", 2:"airplane"}
for i, name in enumerate(COCO_CLASSES):
    names[i+3] = name

data = {
    "path": OUTPUT,
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/test",
    "nc": 12,
    "names": names
}
with open(f"{OUTPUT}/data.yaml", "w") as f:
    yaml.dump(data, f, default_flow_style=False)

print(f"Dataset ready!")

# ── TRAIN ──────────────────────────────────────────────
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data=f"{OUTPUT}/data.yaml",
    epochs=50,
    imgsz=416,
    batch=2,
    device="cpu",
    project="/Users/rutveesureja/Documents/DroneProject/runs",
    name="final_model",
    patience=10,
    pretrained=True,
    freeze=10,
    lr0=1e-4,
    workers=0,
    cache=False,
    plots=True
)
print("Training done!")
