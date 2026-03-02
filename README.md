# Hemoglobin Prediction from Blood Images

A computer vision and machine learning system that predicts hemoglobin levels from blood sample images.

## 📋 Overview

This project uses image processing techniques to extract color features from blood sample images and trains a linear regression model to predict hemoglobin (HB) values.

## 🚀 Features

- Automatic circular region detection in blood images
- LAB color space analysis with lighting normalization
- Linear regression model for HB prediction
- Anemia risk classification (Normal/Mild/Moderate/Severe)
- Batch processing for multiple images

## 📊 Performance

- **MAE**: 0.48 g/dL (average prediction error)
- **R²**: 0.946 (explains 94.6% of variance)

## 🛠️ Installation

```bash
pip install -r requirements.txt