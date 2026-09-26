# TinyML Lab 2: Multi-Class Human Activity Recognition & Model Quantization Scheme Analysis

## Overview
This repository contains the complete implementation for **Lab 2** of the TinyML laboratory course on branch `Lab-2`.

The project implements an end-to-end edge-deployable TinyML pipeline for 3-class **Human Activity Recognition (HAR)** using tri-axial smartphone accelerometer data captured via the **phyphox** mobile application. The target activities include:
- **Idle** (Class 0)
- **Walking** (Class 1)
- **Sit-ups** (Class 2)

---

## Lab Objectives & Pipeline
1. **Sensor Data Acquisition:** Capture continuous 3-axis accelerometer data ($a_x$, $a_y$, $a_z$) for Idle, Walking, and Sit-ups using a smartphone with phyphox.
2. **Signal Preprocessing & Windowing:** Segment continuous time-series accelerometer data into fixed **1-second / 100-sample windows**.
3. **Magnitude Computation:** Calculate the 3D acceleration magnitude for each sample:
   $$|a| = \sqrt{a_x^2 + a_y^2 + a_z^2}$$
4. **Statistical Feature Extraction:** Extract key statistical features from the acceleration magnitude of each window:
   - **Mean**
   - **Standard Deviation**
   - **Root Mean Square (RMS)**
5. **Model Architecture:** Train a lightweight Multi-Layer Perceptron (MLP) classifier for 3-class activity recognition on the extracted feature representations.
6. **TensorFlow Lite Quantization Schemes:** Convert and export the trained baseline model into four distinct TFLite variants:
   - **Float32 Baseline:** Unquantized 32-bit floating point model.
   - **Dynamic Range Quantization:** 8-bit quantized weights with floating-point activations.
   - **Float16 Quantization:** 16-bit floating point weights and computations.
   - **Full Integer (INT8) Quantization:** Fully quantized 8-bit weights and activations using a representative dataset calibration pipeline.
7. **Benchmarking & Trade-off Analysis:** Benchmark and compare all four model formats in terms of:
   - Model file size (bytes / KB)
   - Classification accuracy
   - Memory footprint reduction (%)
   - On-device inference latency per window (ms / µs)

---

## System Architecture & Workflow

![Lab 2 Workflow](Workflow.jpeg)

---

## Repository Structure (Branch: `Lab-2`)

```
├── 2548560_Tejas_R_M_Lab2_HAR_Quantization.ipynb
├── idle.csv
├── walking.csv
├── situps.csv
├── model_float32.tflite
├── model_dynamic_range.tflite
├── model_float16.tflite
├── model_int8.tflite
├── Workflow.jpeg
└── README.md
```

---

## Dataset Description
- `idle.csv`: Raw tri-axial accelerometer recordings during stationary / idle state.
- `walking.csv`: Raw tri-axial accelerometer recordings during continuous walking activity.
- `situps.csv`: Raw tri-axial accelerometer recordings during repetitive sit-up exercises.

---

## Quantized Model Artifacts
- `model_float32.tflite`: Full precision 32-bit floating point TFLite model.
- `model_dynamic_range.tflite`: Post-training dynamically quantized 8-bit weight model.
- `model_float16.tflite`: Half-precision 16-bit floating point TFLite model.
- `model_int8.tflite`: Full integer 8-bit quantized TFLite model for ultra-low-power edge microcontrollers.

---

## Requirements & Prerequisites
To run the notebook locally or in Google Colab, install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## How to Run
1. Checkout the `Lab-2` branch:
   ```bash
   git clone -b Lab-2 https://github.com/Tejas2913/TinyML-Lab-2548560.git
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook 2548560_Tejas_R_M_Lab2_HAR_Quantization.ipynb
   ```
3. Run the notebook cells sequentially with the CSV datasets (`idle.csv`, `walking.csv`, `situps.csv`) in the same working directory.
