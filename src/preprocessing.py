import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_and_preprocess_malimg(raw_dir="data/raw/malimg_paper_dataset_imgs", 
                                target_size=(64, 64), 
                                max_per_class=400, 
                                test_size=0.2, 
                                val_size=0.1):
    """
    Loads Malimg dataset images, handles severe class imbalance by capping 
    oversized families, resizes images, and creates stratified train/val/test splits.
    """
    
    print("[INFO] Scanning dataset directory...")
    class_folders = [f for f in sorted(os.listdir(raw_dir)) if os.path.isdir(os.path.join(raw_dir, f))]
    
    images = []
    labels = []
    class_names = []
    
    for idx, class_name in enumerate(class_folders):
        class_path = os.path.join(raw_dir, class_name)
        img_paths = glob.glob(os.path.join(class_path, "*.png"))
        
        if not img_paths:
            continue
            
        class_names.append(class_name)
        
        # Apply capping to handle extreme class imbalance (e.g., Allaple families)
        if len(img_paths) > max_per_class:
            img_paths = np.random.choice(img_paths, max_per_class, replace=False)
            
        print(f"[LOADING] Class {idx}: {class_name} ({len(img_paths)} images loaded)")
        
        for img_path in img_paths:
            try:
                with Image.open(img_path) as img:
                    img = img.convert('L') # Convert to grayscale
                    img = img.resize(target_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0 # Normalize
                    images.append(img_array)
                    labels.append(idx)
            except Exception as e:
                print(f"[ERROR] Failed to load {img_path}: {e}")
                
    X = np.array(images).reshape(-1, target_size[0], target_size[1], 1)
    y = np.array(labels)
    
    print(f"[INFO] Total dataset shape: X={X.shape}, y={y.shape}")
    
    # Stratified Train / Test Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # Train / Validation Split
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=42
    )
    
    print(f"[INFO] Split complete -> Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Save splits to disk
    os.makedirs("data/splits", exist_ok=True)
    np.save("data/splits/X_train.npy", X_train)
    np.save("data/splits/y_train.npy", y_train)
    np.save("data/splits/X_val.npy", X_val)
    np.save("data/splits/y_val.npy", y_val)
    np.save("data/splits/X_test.npy", X_test)
    np.save("data/splits/y_test.npy", y_test)
    print("[INFO] Preprocessed splits successfully saved to data/splits/!")

if __name__ == "__main__":
    load_and_preprocess_malimg()
