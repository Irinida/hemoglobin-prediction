import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# CONFIGURATION
# =========================
DATASET_PATH = "test_dataset"

# =========================
# FEATURE EXTRACTION
# =========================
def extract_circle_lab(image_path):
    """Extract mean LAB values from circular region in blood sample image"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.resize(img, (300, 300))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect circle (blood sample region)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=50,
        maxRadius=150
    )
    
    # Create mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    if circles is not None:
        x, y, r = circles[0][0]
        x, y, r = int(x), int(y), int(r)
        cv2.circle(mask, (x, y), r, 255, -1)
    else:
        # Fallback: center crop
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        radius = min(center[0], center[1]) - 20
        cv2.circle(mask, center, radius, 255, -1)
    
    # Convert to LAB and normalize lighting
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.equalizeHist(L)  # Normalize lighting
    lab = cv2.merge([L, A, B])
    
    # Extract only circle pixels
    masked_pixels = lab[mask == 255]
    
    if len(masked_pixels) == 0:
        return None
    
    mean_lab = np.mean(masked_pixels, axis=0)
    return mean_lab

# =========================
# LOAD DATA
# =========================
def load_dataset(dataset_path):
    """Load images and extract features with labels"""
    X = []
    y = []
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return np.array(X), np.array(y)
    
    for label_folder in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label_folder)
        
        if not os.path.isdir(label_path):
            continue
        
        # Extract HB value from folder name (e.g., "8gbl" -> 8)
        if "gbl" in label_folder:
            hb_value = int(label_folder.replace("gbl", ""))
        else:
            # Skip folders that don't follow the naming convention
            continue
        
        # Process each image in the folder
        for file in os.listdir(label_path):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(label_path, file)
            features = extract_circle_lab(img_path)
            
            if features is not None:
                X.append(features)
                y.append(hb_value)
                print(f"  Loaded: {file} -> HB={hb_value}")
    
    return np.array(X), np.array(y)

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    print("🔬 Hemoglobin Prediction Model Training")
    print("=" * 50)
    
    # Load dataset
    X, y = load_dataset(DATASET_PATH)
    
    if len(X) == 0:
        print("\n❌ No valid images found. Check your dataset path.")
        exit()
    
    print(f"\n✅ Loaded {len(X)} images")
    print(f"📊 Feature shape: {X.shape}")
    print(f"🏷️ Labels shape: {y.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Predictions
    predictions = model.predict(X_scaled)
    
    # Metrics
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print("\n" + "=" * 50)
    print("📈 REGRESSION RESULTS")
    print("=" * 50)
    print(f"MAE : {mae:.4f} g/dL")
    print(f"R²  : {r2:.4f}")
    
    # Display equation
    a, b, c = model.coef_
    d = model.intercept_
    print("\n📐 Regression Equation:")
    print(f"HB = {a:.4f} × L + {b:.4f} × A + {c:.4f} × B + {d:.4f}")
    
    # Save model
    joblib.dump(model, "hb_regression_model.pkl")
    joblib.dump(scaler, "hb_scaler.pkl")
    print("\n💾 Model saved: hb_regression_model.pkl")
    print("💾 Scaler saved: hb_scaler.pkl")
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(y, predictions, alpha=0.7)
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='Perfect Prediction')
    plt.xlabel("Actual Hemoglobin (g/dL)")
    plt.ylabel("Predicted Hemoglobin (g/dL)")
    plt.title("Actual vs Predicted Hemoglobin")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_results.png", dpi=150)
    plt.show()
    
    print("\n✅ Training complete!")