import os
import shutil
import random

source_dir = "dataset"
dest_dir = "dataset_split"

# Split ratios
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(dest_dir, split), exist_ok=True)

classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

for cls in classes:
    # Create class directories in splits
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)
        
    class_dir = os.path.join(source_dir, cls)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    
    total = len(images)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    train_imgs = images[:train_count]
    val_imgs = images[train_count:train_count + val_count]
    test_imgs = images[train_count + val_count:]
    
    for img in train_imgs:
        shutil.copy(os.path.join(class_dir, img), os.path.join(dest_dir, 'train', cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_dir, img), os.path.join(dest_dir, 'val', cls, img))
    for img in test_imgs:
        shutil.copy(os.path.join(class_dir, img), os.path.join(dest_dir, 'test', cls, img))
        
    print(f"Class {cls}: {total} images -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

print("\nDataset split complete!")
