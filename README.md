# 🛡️ ByteGuard: Automated Malware Family Multi-Classification Pipeline 

> **Transforming Cybersecurity Threat Triage through Computer Vision and Machine Learning**

**ByteGuard** reformulates static malware analysis into a computer vision classification problem. By interpreting compiled Portable Executable (PE) binaries as raw 8-bit grayscale byte matrices, structural similarities across code blocks, import tables, and padding manifest as distinct visual textures. This allows for rapid, **non-execution classification** of malicious software into specific threat families without the need for manual reverse-engineering or risky dynamic sandbox execution.

This project was developed as a **Complex Computing Activity (CCA)** for *CT-264: Programming for Artificial Intelligence* at NED University of Engineering & Technology.

## 📖 Table of Contents

* [The Problem &amp; Our Solution](#-the-problem--our-solution)
* [Key Features](#-key-features)
* [Dataset Acquisition &amp; Engineering](#-dataset-acquisition--engineering)
* [Model Architectures &amp; Performance](#-model-architectures--performance)
* [Repository Architecture](#-repository-architecture)
* [Installation &amp; Quick Start](#-installation--quick-start)
* [Testing &amp; Evaluation Guide](#-testing--evaluation-guide)
* [Team Members](#-team-members)

## ⚡ The Problem & Our Solution

**The Threat:** Traditional signature-based antivirus engines fail against zero-day modifications, polymorphic behavior, and executable packers (like UPX). Dynamic analysis is computationally expensive and vulnerable to sandbox evasion.
**The Solution:** ByteGuard bypasses obfuscation by converting the binary directly into a $64\times64$ grayscale image. Even if the cryptographic hash changes, the macro-level visual texture of the malware family remains consistent, allowing deep learning and ensemble models to identify the threat purely by its visual signature.

## ✨ Key Features

* **Zero-Execution Safety:** Malware binaries are analyzed purely as inert image tensors, eliminating the risk of host infection.
* **Class Imbalance Mitigation:** Deterministic class-capping prevents model bias toward dominant worm families.
* **Multi-Model Intelligence:** Compares 2D spatial convolution (CNN) against high-dimensional decision trees (Random Forest) and geometric hyperplanes (Linear SVM).
* **Professional UI:** A multi-tab Streamlit dashboard providing live threat inference, probability telemetry, and dataset analytics.
* **Modern Tooling:** Built with `uv` for ultra-fast, deterministic, and isolated dependency management.

## 📊 Dataset Acquisition & Engineering

ByteGuard utilizes the benchmark **Malimg (Malware Images) Dataset**, originally introduced by Nataraj et al., comprising **9,339 grayscale `.png` images** across **25 distinct malware families** (Worms, Trojans, Dialers, Rogue Antivirus, Backdoors, and Password Stealers).

### 📥 How to Download the Dataset (Optional)

Because raw dataset binaries and generated `.npy` splits are excluded from version control via `.gitignore` to keep the repository lightweight, you can download the dataset from Kaggle to run full retraining or live image testing:

1. **Download from Kaggle:** [Malimg Dataset (Malware Images)](https://www.kaggle.com/datasets/ikrambenabd/malimg-original) *(or search `"Malimg Malware Image Dataset"` on Kaggle).*
2. **Extract the Archive:** Unzip the downloaded archive and place the `malimg_paper_dataset_imgs` folder (which contains the 25 malware family subdirectories, from `Adialer.C` to `Yuner.A`) inside the `data/raw/` directory:
   ```text
   ByteGuard/data/raw/malimg_paper_dataset_imgs/<25_Malware_Family_Folders>
   ```

*(Note: Pre-trained model checkpoints are already included in `src/models/`, so you can launch the Streamlit dashboard and test individual `.png` malware samples immediately without retraining from scratch!)*

### 🔧 Data Engineering Protocol

* **Class Capping:** To prevent gradient dominance by prolific worm families (`Allaple.A` with 2,949 samples and `Allaple.L` with 1,591 samples), oversized classes are capped at `400` samples per class, yielding a balanced **5,160-image corpus**.
* **Normalization & Resizing:** All images are standardized to `64x64` grayscale resolution (`L` mode) and normalized to a `[0.0, 1.0]` 32-bit floating-point interval.
* **Stratified Partitioning:** The dataset undergoes a stratified **70 / 10 / 20** split (`random_state=42`):
  * **Train (`X_train.npy`):** 3,612 samples
  * **Validation (`X_val.npy`):** 516 samples
  * **Unseen Test (`X_test.npy`):** 1,032 samples

## 🤖 Model Architectures & Performance

Three distinct classification paradigms were engineered and evaluated on the isolated 1,032-record test set:

| Model Architecture                   |  Test Accuracy  |  Weighted F1  | Macro Precision |  Macro Recall  |    Macro F1    | Artifact Size |
| :----------------------------------- | :--------------: | :------------: | :-------------: | :------------: | :------------: | :-----------: |
| **Random Forest (`n=100`)**  | **97.09%** | **0.97** | **0.96** | **0.96** | **0.96** |    7.15 MB    |
| **3-Layer CNN (Keras)**        | **95.83%** | **0.95** | **0.92** | **0.93** | **0.92** |   14.91 MB   |
| **Linear SVM (`LinearSVC`)** | **95.06%** | **0.95** | **0.93** | **0.93** | **0.93** |    0.82 MB    |

*Insight: The 100-Tree Random Forest operating on flattened 4,096-dimensional vectors outperformed the spatial filters of the 3-layer CNN, proving highly adept at isolating exact byte-offset coordinates across PE header layouts.*

## 🏗️ Repository Architecture

```text
ByteGuard/
├── app/                              # Streamlit Intelligence UI
│   ├── components/                   # Extensible UI widget directory
│   ├── main.py                       # Multi-tab dashboard entry point
│   └── styling.css                   # Custom CSS layout & theme rules
├── data/
│   ├── raw/                          # Place malimg_paper_dataset_imgs/ here (25 classes)
│   └── splits/                       # Generated 70/10/20 NumPy arrays (.npy)
├── docs/                             # Technical reports and high-DPI EDA assets
│   ├── assets/
│   │   ├── class_distribution.png    # 300 DPI training distribution plot
│   │   └── sample_images.png         # 300 DPI 64x64 malware texture grid
│   └── CCA_Technical_Report.docx     # Comprehensive CCA technical report
├── notebooks/
│   └── 01_exploratory_analysis.ipynb # Interactive Jupyter EDA notebook
├── src/                              # Modular Python backend
│   ├── preprocessing.py              # Class capping, resizing, and stratified splitting
│   ├── eda_visuals.py                # Automated high-DPI EDA chart generator
│   └── models/                       # Training scripts & serialized model checkpoints
│       ├── cnn_classifier.py         # 3-Layer CNN training & evaluation script
│       ├── rf_classifier.py          # Random Forest training & evaluation script
│       ├── svm_classifier.py         # Linear SVM training & evaluation script
│       ├── byteguard_cnn.keras       # Pre-trained 3-Layer CNN checkpoint
│       ├── byteguard_rf.joblib       # Pre-trained 100-Tree Random Forest model
│       └── byteguard_svm.joblib      # Pre-trained LinearSVC baseline model
├── pyproject.toml                    # uv dependencies and project metadata
└── uv.lock                           # Strict lockfile for 100% reproducibility
```

## 🚀 Installation & Quick Start

This project uses [uv](https://github.com/astral-sh/uv) for blazing-fast, deterministic Python environment management (pinned to **Python 3.12** for full TensorFlow compatibility).

**1. Clone the repository and sync dependencies:**

```powershell
git clone https://github.com/RazaJavaid2004/ByteGuard.git
cd ByteGuard

# Automatically creates .venv and installs exact dependencies from uv.lock
uv sync
```

**2. Launch the Threat Intelligence Dashboard Immediately (Using Pre-Trained Models):**

```powershell
uv run streamlit run app/main.py
```

**3. (Optional) Run the Full Data Engineering & Training Pipeline from Scratch:**
*Requires the Malimg dataset extracted to `data/raw/malimg_paper_dataset_imgs/`.*

```powershell
# Step 1: Generate balanced 64x64 train/val/test splits in data/splits/
uv run python src/preprocessing.py

# Step 2: Generate 300-DPI EDA visualizations in docs/assets/
uv run python src/eda_visuals.py

# Step 3: Train and evaluate all three classifiers on the 1,032 unseen test set
uv run python src/models/cnn_classifier.py
uv run python src/models/rf_classifier.py
uv run python src/models/svm_classifier.py
```

## 🧪 Testing & Evaluation Guide

You can test and verify ByteGuard's classification capabilities in two ways:

### Method 1: Live Interactive Single-Image Testing (Streamlit UI)

1. Run `uv run streamlit run app/main.py` and open the local URL (`http://localhost:8501`) in your browser.
2. Navigate to **Tab 1 (`🔍 Threat Classifier`)**.
3. Click **Browse files** and select any `.png` malware image from any of the 25 family folders inside `data/raw/malimg_paper_dataset_imgs/<Family_Name>/`.
4. **Inspect the Multi-Model Verdicts:**
   * **CNN Verdict Card:** Outputs the predicted malware family along with the **Softmax Confidence Percentage** bar.
   * **Random Forest Verdict Card:** Outputs the ensemble prediction along with the **Tree-Voting Confidence Percentage** (`predict_proba` across 100 trees).
   * **Linear SVM Verdict Card:** Outputs the geometric hyperplane prediction (*Hard Margin*).
5. Switch to **Tab 2 (`📊 Analytics Dashboard`)** and **Tab 3 (`🤖 Model Comparison`)** to inspect dataset KPIs, class balance charts, and comparative test metrics (Precision, Recall, and F1-Score).

### Method 2: Notable Empirical Test Cases to Try

When testing images from different folders, observe these real-world behavioral insights documented in our study:

* **High-Certainty Families (`100%` Confidence on CNN & RF):** Upload samples from `Adialer.C`, `Agent.FYI`, `Dialplatform.B`, `Dontovo.A`, `Fakerean`, `Instantaccess`, `Obfuscator.AD`, or `VB.AT`. Both models lock onto their rigid PE section headers with ~`100%` certainty.
* **Packer-Induced Overlap (`Autorun.K` vs. `Yuner.A`):** Upload a sample from `Autorun.K`. Notice how **Random Forest (`100%`)** and **Linear SVM** accurately identify `Autorun.K` via exact top-row header pixels, whereas the **CNN** classifies it as `Yuner.A` (`91.17%`) because both families are AutoIT worms packed with identical UPX stubs whose macro-textures merge after `MaxPooling2D`.
* **Polymorphic Sibling Variance (`Swizzor.gen!E` & `C2LOP.gen!g`):** Upload samples from `Swizzor.gen!E` or `C2LOP.gen!g`. All three models predict the **correct** family, even though confidence scores are lower (e.g., `53%–63%` CNN, `31%–41%` RF) due to vote-splitting across sub-variants (`C2LOP.P` and `Swizzor.gen!I`).

## 🎓 Team Members

* **Muhammad Raza** (CT-24138) - *Random Forest & Linear SVM Benchmarking, Live Telemetry Evaluation*
* **Haris Zamir** (CT-24125) - *Dataset Curation, Class Imbalance Mitigation, EDA*
* **Ali Hassan** (CT-24136) - *Environment Architecture, 3-Layer CNN, Streamlit Dashboard*
* **Muhammad Saad Khan** (CT-24278) - *Report Formatting, Quality Assurance, Table Optimization*

> **Institution:** Department of Computer Science & Information Technology, NED University of Engineering and Technology, Karachi.
