import cv2
import numpy as np
import os

# =========================
# CONFIGURATION
# =========================
BASE_IMAGE_PATH = "wet_specimen (1).jpg"
OUTPUT_DIR = "synthetic_dataset"
LEVELS = [6, 8, 10, 12, 14]
REFERENCE_LEVEL = 14  # Assume base image corresponds to this Hb

# Color adjustment parameters (tuned for visual effect)
L_SCALE = 0.02   # L change per g/dL from reference
A_SCALE = 0.05   # A change per g/dL from reference

# =========================
# STEP 1: Load image
# =========================
img = cv2.imread(BASE_IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Base image not found: {BASE_IMAGE_PATH}")
img = cv2.resize(img, (300, 300))

# =========================
# STEP 2: Detect blood circle
# =========================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

mask = np.zeros(img.shape[:2], dtype=np.uint8)
if circles is not None:
    x, y, r = circles[0][0]
    x, y, r = int(x), int(y), int(r)
    cv2.circle(mask, (x, y), r, 255, -1)
    print(f"Circle detected at ({x}, {y}) with radius {r}")
else:
    # Fallback: use center crop
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center[0], center[1]) - 20
    cv2.circle(mask, center, radius, 255, -1)
    print("Circle detection failed; using center crop.")

# =========================
# STEP 3: Compute reference mean LAB of blood region
# =========================
blood_pixels = img[mask == 255]
blood_lab = cv2.cvtColor(blood_pixels.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3)
mean_lab_ref = np.mean(blood_lab, axis=0)
print(f"Reference LAB (Hb={REFERENCE_LEVEL}): L={mean_lab_ref[0]:.1f}, A={mean_lab_ref[1]:.1f}, B={mean_lab_ref[2]:.1f}")

# =========================
# STEP 4: Generate one image per level
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

for level in LEVELS:
    delta = REFERENCE_LEVEL - level  # positive for lower Hb
    # Scale factors: L increases for lower Hb, A decreases
    L_adj = 1.0 + delta * L_SCALE
    A_adj = 1.0 - delta * A_SCALE
    B_adj = 1.0  # keep B unchanged

    # Apply adjustment to the blood region
    lab_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2LAB).astype(np.float32)
    # Adjust each channel within mask
    lab_img[:,:,0][mask == 255] *= L_adj
    lab_img[:,:,1][mask == 255] *= A_adj
    lab_img[:,:,2][mask == 255] *= B_adj
    lab_img = np.clip(lab_img, 0, 255).astype(np.uint8)
    new_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

    # Save image in level folder
    level_dir = os.path.join(OUTPUT_DIR, f"{level}gdl")
    os.makedirs(level_dir, exist_ok=True)
    out_path = os.path.join(level_dir, f"sample_1.jpg")
    cv2.imwrite(out_path, new_img)
    print(f"Saved: {out_path}")

print("\n✅ First images generated! You now have one image per level in 'synthetic_dataset/'.")