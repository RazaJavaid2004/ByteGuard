import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_visuals(splits_dir="data/splits", raw_dir="data/raw/malimg_paper_dataset_imgs", output_dir="docs/assets"):
    print("[INFO] Generating Exploratory Data Analysis visuals...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    X_train = np.load(os.path.join(splits_dir, "X_train.npy"))
    y_train = np.load(os.path.join(splits_dir, "y_train.npy"))
    
    class_names = [f for f in sorted(os.listdir(raw_dir)) if os.path.isdir(os.path.join(raw_dir, f))]
    
    # 1. Class Distribution Plot (Capped)
    plt.figure(figsize=(14, 7))
    sns.countplot(x=y_train, hue=y_train, palette="viridis", legend=False)
    plt.title("Malware Family Distribution in Training Set (Post-Capping)", fontsize=14)
    plt.xlabel("Malware Family", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300)
    plt.close()
    print(f"[INFO] Saved class_distribution.png to {output_dir}/")
    
    # 2. Sample Malware Images Grid
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_train[i].reshape(64, 64), cmap='gray')
        plt.title(class_names[y_train[i]], fontsize=10)
        plt.axis("off")
    plt.suptitle("Sample Malware Visualizations (64x64 Grayscale)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_images.png"), dpi=300)
    plt.close()
    print(f"[INFO] Saved sample_images.png to {output_dir}")
    print("[INFO] EDA Generation Complete.")

if __name__ == "__main__":
    generate_eda_visuals()
