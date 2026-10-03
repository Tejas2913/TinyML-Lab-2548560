# TinyML Lab 4: Quantization-Aware Training (QAT)

## Overview
This repository contains the implementation and benchmarking for **Lab 4** of the TinyML laboratory course.

The project implements an end-to-end edge vision pipeline comparing **Float32 baseline**, **Full INT8 Post-Training Quantization (PTQ)**, and **Quantization-Aware Training (QAT)** for image classification on the **CIFAR-10** dataset using **MobileNetV2** ($\alpha = 0.35$).

The objective is to evaluate the trade-offs in model size (KB), classification accuracy (%), and average inference latency (ms) across the three model variants when targeted for resource-constrained edge hardware.

Edge execution and latency benchmarking are **simulated with the TensorFlow Lite interpreter inside the notebook**. No physical deployment on a microcontroller or edge accelerator is performed.

> **Scope note:** this is an experimental proof-of-concept for edge model optimization. See [Experimental Scope and Limitations](#experimental-scope-and-limitations).

---

## Lab Objectives
1. **Dataset acquisition & preprocessing:** load the CIFAR-10 dataset, extract training/testing subsets, resize images to 96×96 RGB, and apply MobileNetV2 preprocessing.
2. **Compact model construction:** instantiate MobileNetV2 with width multiplier $\alpha = 0.35$ (ImageNet pretrained weights) and attach a custom classification head.
3. **Float32 baseline training:** train and evaluate the full-precision Float32 baseline model on the classification task.
4. **Post-Training Quantization (PTQ):** convert the Float32 model to full INT8 precision using representative dataset calibration and measure quantization loss.
5. **Quantization-Aware Training (QAT):** insert fake-quantization emulation nodes into the Keras model using the TensorFlow Model Optimization Toolkit (`tfmot`).
6. **QAT fine-tuning & conversion:** fine-tune the QAT model with a low learning rate and convert it to a full INT8 TFLite model.
7. **Notebook-based benchmarking:** evaluate test accuracy, file size, and per-sample inference latency for Float32, PTQ INT8, and QAT INT8 using the TFLite interpreter.
8. **Comparative analysis:** analyze how QAT recovers accuracy degradation caused by standard PTQ on low-parameter edge architectures.

---

## System Architecture & Workflow

![Lab 4 Workflow](Lab%204_%20Quantisation-Aware%20Training%20Workflow.png)

Sequence actually implemented in the notebook:

```
CIFAR-10 Dataset
        ↓
Image Preprocessing (Resize to 96×96, Normalization)
        ↓
MobileNetV2 (ImageNet pretrained, α=0.35) + Custom Head
        ↓
Float32 Baseline Training
        ↓
Float32 Evaluation
        ↓
Full INT8 PTQ (Representative Dataset Calibration)
        ↓
PTQ INT8 Evaluation
        ↓
Quantization-Aware Training (tfmot Fake-Quant Nodes)
        ↓
QAT Fine-Tuning (Low Learning Rate)
        ↓
Full INT8 QAT Conversion
        ↓
QAT INT8 Evaluation
        ↓
Float32 vs PTQ INT8 vs QAT INT8 Comparison
```

---

## Dataset Description
- **Dataset:** CIFAR-10 (10 object categories).
- **Classes:** `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.
- **Original format:** 60,000 $32 \times 32$ RGB images (50,000 training, 10,000 testing).
- **Actual subset used in the notebook:**
  - **Training subset:** 5,000 randomly sampled images.
  - **Test subset:** 1,000 randomly sampled images.
- **Pixel values:** 8-bit unsigned integers (`uint8`, range $[0, 255]$).

---

## Data Preprocessing
The images are prepared for transfer learning with MobileNetV2 through the following pipeline:
1. **Spatial resizing:** images are upsampled/resized from $32 \times 32$ to $96 \times 96$ pixels using bilinear interpolation to match MobileNetV2's minimum input resolution requirements.
2. **Channel normalization:** input pixel values are scaled to the range $[-1.0, 1.0]$ using MobileNetV2 standard preprocessing (`tf.keras.applications.mobilenet_v2.preprocess_input`).
3. **Data pipeline optimization:** training, validation, and test datasets are structured with `tf.data`, applying batching (`batch_size = 32`) and prefetching (`tf.data.AUTOTUNE`) for efficient execution.

---

## Model
The vision backbone is **MobileNetV2** with a reduced width multiplier ($\alpha = 0.35$), making it exceptionally lightweight and suited for edge deployment:
- **Base model:** MobileNetV2 initialized with ImageNet pre-trained weights.
- **Input shape:** $(96, 96, 3)$.
- **Classification head:**
  - `GlobalAveragePooling2D`
  - `Dropout(0.2)`
  - `Dense(10, activation='softmax')`

**Model Flow:**
$$\text{Input Image } (96 \times 96 \times 3) \longrightarrow \text{MobileNetV2 Backbone } (\alpha = 0.35) \longrightarrow \text{GlobalAveragePooling2D} \longrightarrow \text{Dropout } (0.2) \longrightarrow \text{Dense } (10, \text{Softmax})$$

---

## Float32 Baseline
The baseline Float32 model serves as the reference full-precision network trained with Categorical Crossentropy loss and Adam optimizer.

**Baseline Metrics:**
- **Model Size:** 1596.96 KB
- **Test Accuracy:** 71.30%
- **Average Latency:** 0.475 ms

---

## Post-Training Quantization (PTQ)
Post-Training Quantization converts the trained Float32 model directly into 8-bit integers without additional training epochs:
- **Converter:** `tf.lite.TFLiteConverter.from_keras_model`.
- **Optimization flag:** `tf.lite.Optimize.DEFAULT`.
- **Calibration:** a representative dataset generator provides sample images to calibrate activation dynamic ranges.
- **Integer constraints:** input and output tensors are enforced to `tf.int8`.

**PTQ Metrics:**
- **Model Size:** 609.98 KB
- **Test Accuracy:** 46.30%
- **Average Latency:** 0.758 ms

> **Key Observation:** Direct full INT8 PTQ caused a severe **25.00% accuracy drop** (71.30% down to 46.30%) due to quantization error accumulation in the small $\alpha = 0.35$ parameter space.

---

## Quantization-Aware Training (QAT)
Quantization-Aware Training addresses PTQ degradation by simulating 8-bit quantization noise during the forward pass while keeping gradient updates in floating point:
- **Toolkit:** TensorFlow Model Optimization Toolkit (`tfmot.quantization.keras.quantize_model`).
- **Graph modification:** inserts fake-quantization nodes into layers to model clamping and rounding.
- **Fine-tuning:** fine-tuned for 10 epochs using a reduced learning rate ($1 \times 10^{-4}$).

**QAT Fine-Tuning Progression:**
- **Best Validation Accuracy:** **72.20%** at Epoch 6.
- **QAT Keras Test Accuracy:** **72.80%**.
- **QAT INT8 TFLite Test Accuracy:** **73.00%**.

**Final QAT TFLite Metrics:**
- **Model Size:** 616.97 KB
- **Test Accuracy:** 73.00%
- **Average Latency:** 0.748 ms

---

## TFLite Conversion
The experimental workflow produces three standalone TensorFlow Lite flatbuffer files:

| File | Precision | Purpose |
|---|---|---|
| `model_float32.tflite` | 32-bit Float | Full-precision baseline reference. |
| `model_ptq_int8.tflite` | 8-bit Integer | Post-training quantized model (offline calibration). |
| `model_qat_int8.tflite` | 8-bit Integer | Quantization-aware fine-tuned model. |

---

## Evaluation
All three TFLite models are evaluated on the identical 1,000-sample CIFAR-10 test set using the TensorFlow Lite Interpreter inside the notebook:
- **Size:** measured from the serialized `.tflite` flatbuffer on disk (KB).
- **Accuracy:** percentage of correct top-1 classifications over the test subset.
- **Latency:** average single-sample inference time (ms) computed over test inferences.

*Note: Latency measurements reflect notebook CPU interpreter execution and simulate relative inference performance.*

---

## Experimental Results

| Model | Size (KB) | Test Accuracy (%) | Avg Latency (ms) |
|---|---:|---:|---:|
| **Float32 Baseline** | 1596.96 | 71.30 | 0.475 |
| **PTQ Full INT8** | 609.98 | 46.30 | 0.758 |
| **QAT Full INT8** | 616.97 | 73.00 | 0.748 |

### QAT Training Summary
- **Best validation accuracy:** 72.20% (Epoch 6)
- **QAT Keras test accuracy:** 72.80%
- **QAT INT8 TFLite test accuracy:** 73.00%

---

## Key Observations
1. **Significant Memory Footprint Reduction:** Both INT8 models achieved an approximate **61.4% reduction in storage size** (~610–617 KB vs. 1596.96 KB for Float32), satisfying typical microcontroller Flash memory budgets.
2. **Severe Accuracy Loss with Standard PTQ:** Direct PTQ caused an unacceptable accuracy collapse to 46.30% (-25.00% relative to baseline), highlighting the fragility of compact depthwise separable layers under naive quantization.
3. **Full Accuracy Recovery via QAT:** Quantization-Aware Training completely eliminated quantization loss, achieving **73.00% test accuracy** (+1.70% over Float32 and +26.70% over PTQ).
4. **Size Parity Between Quantization Schemes:** The QAT INT8 model (616.97 KB) retains the compact size of PTQ INT8 (609.98 KB) while delivering production-grade classification accuracy.

---

## Experimental Scope and Limitations
- **Dataset scale:** evaluated on a representative 5,000 train / 1,000 test subset of CIFAR-10.
- **Architectural scope:** results reflect MobileNetV2 with $\alpha = 0.35$ and custom head configuration.
- **Host simulation:** latency was benchmarked using the TFLite Python Interpreter on a host workstation, not a physical ARM Cortex-M or RISC-V edge microcontroller.
- **Sensitivity:** quantization outcomes depend on the representative calibration set and fine-tuning schedule.

---

## Repository Structure
```
├── 2548560_Tejas_R_M_TinyML_Lab4_QAT_CIFAR10_MobileNetV2.ipynb
├── model_float32.tflite
├── model_ptq_int8.tflite
├── model_qat_int8.tflite
├── Lab 4_ Quantisation-Aware Training Workflow.png
└── README.md
```

---

## Requirements & Prerequisites
The experiment requires Python 3.8+ with the following packages:
- `tensorflow`
- `tensorflow-model-optimization`
- `tf-keras`
- `numpy`
- `pandas`
- `matplotlib`

---

## How to Run
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/Tejas2913/TinyML-Lab-2548560.git
   cd TinyML-Lab-2548560
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow tensorflow-model-optimization tf-keras numpy pandas matplotlib
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook 2548560_Tejas_R_M_TinyML_Lab4_QAT_CIFAR10_MobileNetV2.ipynb
   ```
4. Run all cells sequentially to train the Float32 model, perform PTQ calibration, execute QAT fine-tuning, export TFLite models, and generate the comparative benchmark results.
