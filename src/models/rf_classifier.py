import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def build_and_train_rf(model_save_path="byteguard_rf.joblib"):
    print("[INFO] Loading preprocessed data for Random Forest...")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    splits_dir = os.path.join(base_dir, "data", "splits")
    
    X_train = np.load(os.path.join(splits_dir, "X_train.npy"))
    y_train = np.load(os.path.join(splits_dir, "y_train.npy"))
    X_test = np.load(os.path.join(splits_dir, "X_test.npy"))
    y_test = np.load(os.path.join(splits_dir, "y_test.npy"))
    
    # Flatten the 64x64x1 images for scikit-learn (4096 features)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print(f"[INFO] Flattened training data shape: {X_train_flat.shape}")
    print("[INFO] Training Random Forest Classifier (n_estimators=100)...")
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_flat, y_train)
    
    print("[INFO] Evaluating on Test Set...")
    y_pred = rf_model.predict(X_test_flat)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"[RESULT] Random Forest Test Accuracy: {test_acc:.4f}")
    print("\n[INFO] RF Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_save_path)
    joblib.dump(rf_model, save_path)
    print(f"[INFO] Model saved to {save_path}")

if __name__ == "__main__":
    build_and_train_rf()
