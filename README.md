# Hybrid CNN-Transformer Model for Glaucoma Diagnosis from AS-OCT Images

## Abstract
This repository contains the official PyTorch implementation of the paper: **"Potential role of iris features for classification of angle closure glaucoma: insights from deep-learning-based analysis of AS-OCT scans and study of iris smoothness"**.
We propose a hybrid deep learning framework combining **EfficientNet-B3** and **Swin Transformer** to classify Angle Closure Glaucoma subtypes (PACS, PAC, PACG) and Normal eyes using Anterior Segment OCT (AS-OCT) images. Additionally, a **U-Net** based segmentation module is used for Region of Interest (ROI) extraction.

##  Performance
Our proposed hybrid model achieved state-of-the-art results on the dataset:
- **Classification Accuracy:** 92.88%
- **AUC:** 0.98
- **Segmentation IoU:** 0.8581

| Model | Accuracy | F1 Score | AUC |
| :--- | :---: | :---: | :---: |
| **Hybrid (Ours)** | **0.9288** | **0.9289** | **0.9844** |
| EfficientNet-B3 | 0.8531 | 0.8541 | 0.9624 |
| Swin Transformer | 0.8481 | 0.8504 | 0.9753 |

*Reference: Table 4 and Supplementary Table 3 of the manuscript.*

## Methodology
The pipeline consists of two main stages:
1.  **Iris Segmentation:** A U-Net model extracts the iris region to remove noise.
2.  **Classification:** A hybrid architecture fuses local features (CNN) and global representations (Transformer).

![Workflow](results/figures/workflow_diagram.jpg)
*Figure 2: The proposed preprocessing and classification pipeline.*

## Project Structure
```text
├── data/                  # Place your dataset here
├── models/                # Model architectures (Hybrid & U-Net)
├── utils/                 # Helper functions (Dataloaders, Metrics)
├── train_classifier.py    # Main script for training the classification model
├── train_segmentation.py  # Main script for training the segmentation model
└── requirements.txt       # Dependencies
