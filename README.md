# Multi-Agent Pneumonia Detection with Explainable AI

An explainable deep learning framework for pneumonia detection from chest X-ray images using CNN and ResNet-18 architectures, combined with Grad-CAM, SHAP, quantitative explanation validation, and an LLM-based multi-agent reasoning workflow.

## Project Overview

This project investigates deep learning-based pneumonia classification using chest X-ray images. The workflow combines predictive modeling with explainable AI and multi-agent reasoning to evaluate both model predictions and generated explanations.

## Key Features

- CNN-based pneumonia classification
- ResNet-18 transfer learning
- Leakage-safe train/validation/test evaluation
- Duplicate-image detection
- Class imbalance handling
- ROC-AUC and classification performance analysis
- Calibration analysis
- Grad-CAM visual explanations
- SHAP-based qualitative explanations
- Explanation validation with a Critic Agent
- Perturbation sensitivity analysis
- LLM-based multi-agent reasoning workflow

## Dataset

Chest X-Ray Images (Pneumonia) dataset from Kaggle.

The dataset is downloaded automatically through the Kaggle API. Kaggle credentials are stored using Google Colab Secrets and are not included in this repository.

## Explainability

The project uses:

- Grad-CAM for spatial explanation
- SHAP for qualitative pixel-level interpretation
- Explanation validation metrics
- Perturbation-based sensitivity analysis

## Multi-Agent Architecture

The implemented workflow includes multiple specialized agents:

- **Prediction Agent** — CNN/ResNet-18 based pneumonia classification
- **Explanation Agent** — Generates Grad-CAM and SHAP explanations for model decisions
- **Critic Agent** — Validates explanation quality through perturbation sensitivity analysis
- **Report Agent** — Synthesizes predictions and explanations into structured clinical summaries via LLM

## Technologies

- Python
- PyTorch
- torchvision
- scikit-learn
- NumPy
- pandas
- Matplotlib
- Grad-CAM
- SHAP
- Google Colab
- Gemini API

## Notebook

The complete experimental pipeline is available in:

`multi_agent_pneumonia_xai.ipynb`
