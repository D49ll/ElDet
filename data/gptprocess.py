import os
import json
import random
from PIL import Image

# -------------------------------
# Configuration
# -------------------------------
# Root folder for your dataset (adjust as needed)
DATA_ROOT = "dataset"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
GT_DIR = os.path.join(DATA_ROOT, "gt")
# Output folder for the JSON files (inside your data folder, under annotations)
ANNOTATIONS_DIR = os.path.join(DATA_ROOT, "annotations")
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# -------------------------------
# Utility: Auto-increment generator
# -------------------------------
def auto_increment_generator(start=1):
    i = start
    while True:
        yield i
        i += 1

image_id_gen = auto_increment_generator(1)
annotation_id_gen = auto_increment_generator(1)

# -------------------------------
# Build unified lists for images and annotations
# -------------------------------
images_list = []
annotations_list = []

# List all image files (assuming .jpg or .png)
image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.png'))])

for img_file in image_files:
    img_path = os.path.join(IMAGES_DIR, img_file)
    
    # Open the image to get dimensions
    try:
        with Image.open(img_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"Error opening image {img_file}: {e}")
        continue

    # Assign an ID to the image and create an image entry
    image_id = next(image_id_gen)
    images_list.append({
        "id": image_id,
        "file_name": img_file,
        "width": width,
        "height": height
    })

    # Determine the corresponding ground truth file.
    # Expected annotation filename: "gt_" + image filename + ".txt"
    gt_filename = f"gt_{img_file}.txt"
    gt_path = os.path.join(GT_DIR, gt_filename)
    if not os.path.exists(gt_path):
        print(f"Annotation file {gt_filename} not found for image {img_file}. Skipping annotations for this image.")
        continue

    # Read annotation file.
    # Format: first line is number of ellipses; subsequent lines each contain:
    # cx  cy  a  b  θ   (values separated by whitespace)
    with open(gt_path, "r") as f:
        lines = f.readlines()
    if not lines:
        print(f"No content in {gt_filename}.")
        continue
    try:
        num_ellipses = int(lines[0].strip())
    except ValueError:
        print(f"First line in {gt_filename} is not an integer. Skipping file.")
        continue

    # Process each ellipse annotation (assume exactly num_ellipses lines follow)
    for line in lines[1:1+num_ellipses]:
        parts = line.strip().split()
        if len(parts) < 5:
            print(f"Malformed annotation in {gt_filename}: {line}")
            continue
        try:
            cx = float(parts[0])
            cy = float(parts[1])
            a  = float(parts[2])
            b  = float(parts[3])
            theta = float(parts[4])
        except ValueError:
            print(f"Error parsing values in {gt_filename}: {line}")
            continue

        annotations_list.append({
            "id": next(annotation_id_gen),
            "image_id": image_id,
            "bbox": [cx, cy, a, b, theta],
            "iscrowd": 0,
            "category_id": 1  # Add the category id; for a single category "ellipse"
        })

# -------------------------------
# Define categories list (for example, one category: "ellipse")
# -------------------------------
categories = [{
    "id": 1,
    "name": "ellipse"
}]

# -------------------------------
# Split into train and test (80/20)
# -------------------------------
all_image_ids = [img["id"] for img in images_list]
random.shuffle(all_image_ids)
split_idx = int(0.8 * len(all_image_ids))
train_ids = set(all_image_ids[:split_idx])
test_ids  = set(all_image_ids[split_idx:])

train_images = [img for img in images_list if img["id"] in train_ids]
test_images  = [img for img in images_list if img["id"] in test_ids]
train_annotations = [ann for ann in annotations_list if ann["image_id"] in train_ids]
test_annotations  = [ann for ann in annotations_list if ann["image_id"] in test_ids]

# -------------------------------
# Build final JSON dictionaries
# -------------------------------
# Including the "categories" field
train_json = {
    "images": train_images,
    "annotations": train_annotations,
    "categories": categories
}
test_json = {
    "images": test_images,
    "annotations": test_annotations,
    "categories": categories
}

# -------------------------------
# Save JSON files
# -------------------------------
train_json_path = os.path.join(ANNOTATIONS_DIR, "train.json")
test_json_path  = os.path.join(ANNOTATIONS_DIR, "test.json")
with open(train_json_path, "w") as f:
    json.dump(train_json, f, indent=4)
with open(test_json_path, "w") as f:
    json.dump(test_json, f, indent=4)

print(f"Total images processed: {len(images_list)}")
print(f"Train: {len(train_images)} images, {len(train_annotations)} annotations")
print(f"Test: {len(test_images)} images, {len(test_annotations)} annotations")
print(f"Train JSON saved to: {train_json_path}")
print(f"Test JSON saved to: {test_json_path}")
