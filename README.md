# Explainable Lung Cancer Classification from CT Images

## Project Overview

This project performs lung cancer CT image classification using transfer learning and explainable deep learning.

The system classifies lung CT images into three classes:

- Normal
- Benign
- Malignant

The project compares multiple deep learning models and deploys the best model using Streamlit.

## Dataset

The project uses the IQ-OTH/NCCD lung cancer CT dataset.

Dataset classes:

- Normal
- Benign
- Malignant

## Models Compared

- ResNet50
- DenseNet121
- EfficientNetB0
- Fine-tuned ResNet50

## Final Selected Model

ResNet50 was selected for deployment because it achieved the highest Macro F1 score.

Macro F1 was used because the dataset is imbalanced and the benign class has fewer samples.

## Main Results

| Model | Accuracy | Macro F1 | Macro ROC-AUC |
|---|---:|---:|---:|
| ResNet50 | 0.8364 | 0.7150 | 0.9064 |
| DenseNet121 | 0.7939 | 0.6932 | 0.9208 |
| ResNet50 Fine-Tuned | 0.8424 | 0.6502 | 0.9472 |
| EfficientNetB0 | 0.5152 | 0.4953 | 0.9009 |

## Explainability

Grad-CAM is used to visualize which regions of the CT image influenced the model prediction.

The app shows:

- Uploaded CT image
- Predicted class
- Prediction probabilities
- Grad-CAM heatmap
- Grad-CAM overlay
- Experimental reliability score

## Project Novelty

This project does not only report accuracy. It compares models using Macro F1 and Grad-CAM-based reliability analysis.

A key finding is that fine-tuning improved overall accuracy but reduced minority-class benign performance. This shows that accuracy alone can be misleading for imbalanced medical datasets.

## Streamlit App Features

- CT image upload
- Lung cancer class prediction
- Probability visualization
- Grad-CAM heatmap
- Model comparison dashboard
- Medical disclaimer

## Deployment

The Streamlit deployment link is: https://explainable-lung-cancer-classification-l3.streamlit.app/
