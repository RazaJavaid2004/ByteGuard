import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

def build_and_train_svm(model_save_path="byteguard_svm.joblib"):
    print("[INFO] Loading preprocessed data for SVM...")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    splits_dir = os.path.join(base_dir, "data", "splits")
    
    X_train = np.load(os.path.join(splits_dir, "X_train.npy"))
    y_train = np.load(os.path.join(splits_dir, "y_train.npy"))
    X_test = np.load(os.path.join(splits_dir, "X_test.npy"))
    y_test = np.load(os.path.join(splits_dir, "y_test.npy"))
    
    # Flatten images
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print("[INFO] Training Linear SVM Classifier...")
    # Using LinearSVC for faster convergence on high-dimensional image data
    svm_model = LinearSVC(random_state=42, max_iter=2000, dual=False)
    svm_model.fit(X_train_flat, y_train)
    
    print("[INFO] Evaluating on Test Set...")
    y_pred = svm_model.predict(X_test_flat)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"[RESULT] SVM Test Accuracy: {test_acc:.4f}")
    print("\n[INFO] SVM Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_save_path)
    joblib.dump(svm_model, save_path)
    print(f"[INFO] Model saved to {save_path}")

if __name__ == "__main__":
    build_and_train_svm()
