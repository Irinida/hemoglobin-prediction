import os
import cv2
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error

# =========================
# LOAD MODEL
# =========================
def load_model():
    try:
        model = joblib.load("hb_regression_model.pkl")
        scaler = joblib.load("hb_scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        print("❌ Model files not found. Run main.py first.")
        exit()

# =========================
# FEATURE EXTRACTION
# =========================
def extract_circle_lab(image_path):
    """Extract mean LAB values from circular region"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.resize(img, (300, 300))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1, minDist=100,
        param1=50, param2=30,
        minRadius=50, maxRadius=150
    )
    
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    if circles is not None:
        x, y, r = circles[0][0]
        x, y, r = int(x), int(y), int(r)
        cv2.circle(mask, (x, y), r, 255, -1)
    else:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        radius = min(center[0], center[1]) - 20
        cv2.circle(mask, center, radius, 255, -1)
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.equalizeHist(L)
    lab = cv2.merge([L, A, B])
    
    masked_pixels = lab[mask == 255]
    
    if len(masked_pixels) == 0:
        return None
    
    return np.mean(masked_pixels, axis=0)

# =========================
# ANEMIA RISK
# =========================
def anemia_risk(hb_value):
    if hb_value >= 12:
        return "🟢 Normal"
    elif hb_value >= 10:
        return "🟡 Mild Anemia Risk"
    elif hb_value >= 8:
        return "🟠 Moderate Anemia Risk"
    else:
        return "🔴 Severe Anemia Risk"

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("🔬 Batch Hemoglobin Prediction")
    print("=" * 50)
    
    model, scaler = load_model()
    
    test_folder = input("Enter test folder path: ").strip()
    
    if not os.path.exists(test_folder):
        print(f"❌ Folder not found: {test_folder}")
        exit()
    
    print(f"\n🔍 Testing images in: {test_folder}")
    print("-" * 50)
    
    y_true = []
    y_pred = []
    
    for root, dirs, files in os.walk(test_folder):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            image_path = os.path.join(root, file)
            features = extract_circle_lab(image_path)
            
            if features is None:
                print(f"⚠️ Skipping {file} (feature extraction failed)")
                continue
            
            features_scaled = scaler.transform([features])
            predicted_hb = model.predict(features_scaled)[0]
            
            # Extract actual HB from folder name
            folder_name = os.path.basename(root)
            if "gbl" in folder_name:
                actual_hb = int(folder_name.replace("gbl", ""))
                
                y_true.append(actual_hb)
                y_pred.append(predicted_hb)
                
                error = abs(actual_hb - predicted_hb)
                risk = anemia_risk(predicted_hb)
                
                print(f"📁 {folder_name} | {file}")
                print(f"  Actual: {actual_hb} g/dL | Predicted: {predicted_hb:.2f} g/dL")
                print(f"  Error: {error:.2f} | {risk}\n")
            else:
                # If folder doesn't follow naming, just predict without actual
                print(f"📁 {folder_name} | {file}")
                print(f"  Predicted: {predicted_hb:.2f} g/dL | {anemia_risk(predicted_hb)}\n")
    
    if len(y_true) > 0:
        mae = mean_absolute_error(y_true, y_pred)
        print("=" * 50)
        print(f"📊 Overall Test MAE: {mae:.3f} g/dL")
    else:
        print("\n⚠️ No valid test images found (folders must contain 'gbl' to compare with actual values).")
    
    print("\n✅ Testing complete.")