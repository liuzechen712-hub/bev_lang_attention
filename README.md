# 🚗 Language-Guided BEVFormer for 3D Object Detection

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![MMDetection3D](https://img.shields.io/badge/mmdet3d-1.0+-green.svg)](https://github.com/open-mmlab/mmdetection3d)

## 📖 Overview
This project explores integrating **natural language instructions** into Bird's-Eye-View (BEV) 3D object detection. Based on **BEVFormer**, we design a plug-and-play **Language-Guided BEV Attention** mechanism (DistilBERT encoder + Cross-Attention fusion) to align linguistic semantics with BEV spatial features.

The repository provides a complete engineering pipeline: data cleaning, source-level model modification, robust training, and feature visualization.

## 📁 Directory Structure
```text
├── projects/       # Core architecture (modified BEVFormer + language modules)
├── tools/          # Training / testing entry scripts
├── scripts/        # Utility scripts (loss plotting, feature visualization)
├── assets/         # Figures for README & report
├── docs/           # Internship project report
└── README.md

