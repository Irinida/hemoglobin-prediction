import os
import cv2
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================
def load_model():
    """Load trained model and scaler"""
    try:
        model = joblib.load("hb_regression_model.pkl")
        scaler = joblib.load("hb_scaler.pkl")
        print("✅ Model loaded successfully")
        return model, scaler
    except FileNotFoundError:
        print("❌ Model files not found. Run main.py first to train the model.")
        exit()

# =========================
# FEATURE EXTRACTION
# =========================
def extract_circle_lab(image_path):
    """Extract mean LAB values from circular region"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
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
    
    # LAB conversion with lighting normalization
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.equalizeHist(L)
    lab = cv2.merge([L, A, B])
    
    masked_pixels = lab[mask == 255]
    
    if len(masked_pixels) == 0:
        return None
    
    return np.mean(masked_pixels, axis=0)

# =========================
# ANEMIA RISK ASSESSMENT
# =========================
def anemia_risk(hb_value):
    """Classify anemia risk based on hemoglobin level"""
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
    print("🔬 Hemoglobin Prediction from Blood Image")
    print("=" * 50)
    
    # Load model
    model, scaler = load_model()
    
    # Get image path
    image_path = input("Enter path to blood sample image: ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        exit()
    
    # Extract features
    print("\n📸 Processing image...")
    features = extract_circle_lab(image_path)
    
    if features is None:
        print("❌ Feature extraction failed")
        exit()
    
    # Predict
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    
    # Display results
    print("\n" + "=" * 50)
    print("📊 PREDICTION RESULTS")
    print("=" * 50)
    print(f"Predicted Hemoglobin: {prediction:.2f} g/dL")
    print(f"Risk Assessment: {anemia_risk(prediction)}")
    
    # Show LAB values
    print(f"\n📐 Extracted LAB values:")
    print(f"  L (Lightness): {features[0]:.2f}")
    print(f"  A (Green-Red): {features[1]:.2f}")
    print(f"  B (Blue-Yellow): {features[2]:.2f}")