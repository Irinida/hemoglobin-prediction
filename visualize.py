import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib

def visualize_circle_detection(image_path):
    """Visualize the circle detection on the image"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1, minDist=100,
        param1=50, param2=30,
        minRadius=50, maxRadius=150
    )
    
    result = img_rgb.copy()
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(result, (x, y), r, (255, 0, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 255, 0), 3)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(result)
    axes[1].set_title("Detected Circle")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        visualize_circle_detection(sys.argv[1])
    else:
        print("Usage: python visualize.py <image_path>")