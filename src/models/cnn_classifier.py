import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, accuracy_score

def build_and_train_cnn(splits_dir="../../data/splits", model_save_path="byteguard_cnn.keras"):
    print("[INFO] Loading preprocessed data...")
    # Adjust paths based on execution directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    splits_dir = os.path.join(base_dir, "data", "splits")
    
    X_train = np.load(os.path.join(splits_dir, "X_train.npy"))
    y_train = np.load(os.path.join(splits_dir, "y_train.npy"))
    X_val = np.load(os.path.join(splits_dir, "X_val.npy"))
    y_val = np.load(os.path.join(splits_dir, "y_val.npy"))
    X_test = np.load(os.path.join(splits_dir, "X_test.npy"))
    y_test = np.load(os.path.join(splits_dir, "y_test.npy"))
    
    num_classes = len(np.unique(y_train))
    
    print("[INFO] Building CNN architecture...")
    model = models.Sequential([
        layers.InputLayer(shape=(64, 64, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("[INFO] Training CNN...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    
    print("[INFO] Evaluating on Test Set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] CNN Test Accuracy: {test_acc:.4f}")
    
    # Generate detailed metrics
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\n[INFO] CNN Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save Model
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_save_path)
    model.save(save_path)
    print(f"[INFO] Model saved to {save_path}")

if __name__ == "__main__":
    build_and_train_cnn()
