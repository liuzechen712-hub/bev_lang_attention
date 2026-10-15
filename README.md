# Language-Guided BEVFormer for 3D Object Detection

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

## Overview
This project integrates natural language instructions into Bird-Eye-View (BEV) 3D object detection. Based on BEVFormer, we design a plug-and-play Language-Guided BEV Attention mechanism (DistilBERT encoder + Cross-Attention fusion) to align linguistic semantics with BEV spatial features.

The repository provides a complete engineering pipeline: data cleaning, source-level model modification, robust training, and feature visualization.

## Directory Structure
- projects/  Core architecture (modified BEVFormer + language modules)
- tools/     Training / testing entry scripts
- scripts/   Utility scripts (loss plotting, feature visualization)
- assets/    Figures for README and report
- docs/      Internship project report

## Quick Start
conda activate bev_drivelm
export PYTHONPATH=$(pwd):$PYTHONPATH
python tools/train.py projects/configs/bevformer/bevformer_standalone_test.py --work-dir work_dirs/bevformer_enhanced --gpus 1 --no-validate
python scripts/plot_loss_zoom.py
python scripts/visualize_features.py

## Experimental Results

### Training Convergence
Stable loss descent in the initial phase validates the cross-modal fusion pipeline.


### Deep Feature Activation Map
The backbone focuses on semantically critical regions while suppressing background.


## Report
See docs/internship_report.md for methodology, engineering challenges and analysis.

## Citation
Li, Zhiqi et al. BEVFormer: Learning Birds-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers. arXiv:2203.17270, 2022.
