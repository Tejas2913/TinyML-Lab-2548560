# TinyML Lab 1: Real-Time Rotational Dynamics & Edge-AI Payload Safety System

## Overview
This repository contains the complete implementation for **Lab 1** of the TinyML laboratory course.

The project implements an end-to-end edge-deployable TinyML pipeline that classifies the rotational stability of a payload resting on a rotating platform as either **SAFE (0)** or **UNSAFE / FALL / SLIDE / TIP (1)** using smartphone gyroscope data ($X$ and $Y$ axes) collected via the **phyphox** application.

---

## Lab Objectives
1. **Sensor Data Acquisition:** Capture real-time rotational dynamics using smartphone IMU gyroscope sensors via phyphox.
2. **Feature Engineering & Preprocessing:** Restrict to Gyroscope $X$ and $Y$ axes, clean noise, window the continuous time-series, and extract key statistical features (Mean, Std, RMS).
3. **Lightweight Classification:** Train a Decision Tree classifier to distinguish between stable (SAFE) and unstable (UNSAFE) payload states based on physical threshold dynamics.
4. **Edge Optimization:** Convert the trained model pipeline into TensorFlow Lite format, generating both Float32 and fully quantized INT8 edge models.
5. **On-Device Simulation & Benchmarking:** Evaluate edge deployment metrics (model size, latency, inference performance) inside the notebook environment.

---

## System Architecture & Workflow

![Lab 1 Workflow](Workflow.png)

### 15-Step Pipeline:
1. **Smartphone + phyphox:** Sensor mounting & setup
2. **Gyroscope Axes Selection:** Focus on $X$ and $Y$ rotational axes
3. **Data Acquisition:** Recording continuous angular velocity
4. **Data Cleaning & Preprocessing:** Timestamp alignment and validation
5. **Time-Series Windowing:** Sliding window segmentation
6. **Feature Extraction:** Statistical metrics over windowed signals
7. **Feature Matrix ($X$):** Multi-dimensional feature representations
8. **Labels ($y$):** Physical ground-truth annotations (0: SAFE, 1: UNSAFE)
9. **Train/Test Split:** Group-aware split for robust validation
10. **Lightweight Classifier:** Decision Tree implementation
11. **Model Evaluation:** Accuracy, Precision, Recall, F1-Score & Confusion Matrix
12. **TensorFlow Lite Conversion:** Optimizing for embedded/microcontroller runtimes
13. **TFLite Models:** Float32 & INT8 quantized artifacts
14. **TFLite Inference:** Simulating on-device execution
15. **Deployment Metrics:** Model size, latency, and memory footprint analysis

---

## Dataset Description
- **File:** `Main_Raw Data.csv`
- **Source:** High-frequency tri-axial gyroscope recordings gathered from the phyphox mobile application during rotational speed ramp-up experiments.
- **Signals Used:** $\omega_x$, $\omega_y$ angular velocity channels across multiple experimental runs and stability transitions.

---

## Models & TFLite Conversion
- **Base Classifier:** Scikit-Learn Decision Tree Classifier.
- **Neural Network / TFLite Target:** Dense neural network architecture built in TensorFlow/Keras matching the extracted feature space.
- **Generated Artifacts:**
  - `model_float32.tflite`: Unquantized 32-bit floating-point TFLite model.
  - `model_int8.tflite`: Post-training 8-bit quantized TFLite model optimized for constrained edge devices and microcontrollers.

---

## Repository Structure (Branch: `Lab-1`)

```
├── 254856_Tejas_R_M_TinyML_LAB01_Real_Time_Rotational_Dynamics_Edge_AI_Payload_Safety_System.ipynb
├── Main_Raw Data.csv
├── model_float32.tflite
├── model_int8.tflite
├── Workflow.png
└── README.md
```

---

## Requirements & Prerequisites
To run the notebook, install the following Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## How to Run
1. Clone or checkout the `Lab-1` branch:
   ```bash
   git clone -b Lab-1 https://github.com/Tejas2913/TinyML-Lab-2548560.git
   ```
2. Open the Jupyter notebook:
   ```bash
   jupyter notebook 254856_Tejas_R_M_TinyML_LAB01_Real_Time_Rotational_Dynamics_Edge_AI_Payload_Safety_System.ipynb
   ```
3. Run the notebook cells sequentially. Ensure `Main_Raw Data.csv` remains in the same working directory.
