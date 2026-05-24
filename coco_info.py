import json

with open('/Users/rutveesureja/fiftyone/coco-2017/train/labels.json') as f:
    coco = json.load(f)

with open('coco_structure.txt', 'w') as out:
    out.write("=== FIRST IMAGE ===\n")
    out.write(json.dumps(coco['images'][0], indent=4))
    
    out.write("\n\n=== FIRST ANNOTATION ===\n")
    out.write(json.dumps(coco['annotations'][0], indent=4))
    
    out.write("\n\n=== FIRST CATEGORY ===\n")
    out.write(json.dumps(coco['categories'][0], indent=4))
    
    out.write("\n\n=== ALL CATEGORIES ===\n")
    for cat in coco['categories']:
        out.write(f"ID: {cat['id']}  →  {cat['name']}\n")
    
    out.write("\n\n=== SUMMARY ===\n")
    out.write(f"Total images: {len(coco['images'])}\n")
    out.write(f"Total annotations: {len(coco['annotations'])}\n")
    out.write(f"Total categories: {len(coco['categories'])}\n")

print("Saved to coco_structure.txt")
