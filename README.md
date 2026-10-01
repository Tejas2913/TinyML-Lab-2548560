# TinyML Lab 1: Real-Time Rotational Dynamics & Edge-AI Payload Safety System

## Overview
This repository contains the implementation for **Lab 1** of the TinyML laboratory course.

The project builds an end-to-end, edge-oriented TinyML pipeline that classifies short windows of smartphone gyroscope data (**X and Y axes**, recorded with the **phyphox** app) as **SAFE (0)** or **UNSAFE / FALL / SLIDE / TIP (1)** for a payload that eventually falls during the experiment.

Two models are built on the same six statistical features:

- a **Decision Tree** (scikit-learn) as the classical ML baseline, and
- a **compact Keras neural network**, which is the model converted to **TensorFlow Lite** (Float32 and INT8).

Edge execution is **simulated with the TensorFlow Lite interpreter inside the notebook**. No physical deployment on a smartphone or microcontroller is performed.

> **Scope note:** this is a small proof-of-concept experiment, not a production benchmark. See [Experimental Scope and Limitations](#experimental-scope-and-limitations).

---

## Lab Objectives
1. **Sensor data acquisition:** record rotational dynamics with a smartphone IMU (gyroscope) via phyphox.
2. **Preprocessing & feature engineering:** use Gyroscope X and Y only, clean and validate the data, window the signal, and extract statistical features (mean, standard deviation, RMS).
3. **Window-size selection:** compare 3.0 s, 1.0 s, 0.5 s and 0.25 s windows (50% overlap) using class coverage, evaluation validity and balanced metrics.
4. **Lightweight classification:** train a Decision Tree baseline using a trial-aware train/test split.
5. **Edge optimization:** train a compact Keras network on the same features and convert it to TensorFlow Lite (Float32 and post-training INT8).
6. **Notebook-based benchmarking:** evaluate TFLite predictions, model size and single-sample inference latency using the TFLite interpreter in the notebook.

---

## System Architecture & Workflow

![Lab 1 Workflow](Workflow.png)

Sequence actually implemented in the notebook:

```
Smartphone + phyphox
        ↓
Gyroscope X/Y acquisition (one continuous recording)
        ↓
Data cleaning and preprocessing
        ↓
Trial segmentation (20 stopwatch-timed fall events)
        ↓
Window-size ablation (3.0 / 1.0 / 0.5 / 0.25 s, 50% overlap)
        ↓
Final windowing with the selected size (1.0 s, 50% overlap)
        ↓
Six statistical features (X/Y mean, std, RMS)
        ↓
SAFE / UNSAFE window labels
        ↓
Trial-aware train/test split (GroupShuffleSplit by trial)
        ↓
Decision Tree baseline
        ↓
Compact Keras neural network (6 → 16 → 8 → 1)
        ↓
Float32 TFLite
        ↓
INT8 post-training quantization
        ↓
TFLite interpreter inference (in notebook)
        ↓
Latency / model-size evaluation
```

---

## Dataset Description
- **File:** `Main_Raw Data.csv` (single continuous phyphox export).
- **Columns in the export:** `Time (s)`, `Gyroscope x (rad/s)`, `Gyroscope y (rad/s)`, `Gyroscope z (rad/s)`, `Absolute (rad/s)`.
- **Signals used:** only Gyroscope **X** and **Y**. Z is detected but is not used anywhere in the pipeline.
- **Size:** 35,846 raw samples over 71.669 s (about 500 Hz, median Δt ≈ 0.002 s).
- **Trials:** the gyroscope was never stopped between falls, so the 20 trials are 20 segments of one continuous signal. They are located by matching 20 independently recorded **stopwatch fall times** to the nearest gyroscope timestamp.
- **Segmentation:** each trial spans ±3.0 s around its fall time, clamped at the midpoint to neighbouring events so trials never overlap. This gives 34,761 samples across 20 trials. The segmentation is **fixed** and is independent of the ML window size, so only the window length changes during the ablation.
- **Data quality:** no missing values, no duplicate rows, no non-numeric entries were found.

---

## Preprocessing & Feature Engineering
Each window is described by six features computed from the gyroscope X and Y angular-velocity samples inside that window. These are the **only classifier inputs** for both the Decision Tree and the Keras network:

| Feature | Description |
|---|---|
| `X_mean` | Mean of gyroscope X in the window |
| `X_std` | Standard deviation of gyroscope X |
| `X_rms` | Root-mean-square of gyroscope X |
| `Y_mean` | Mean of gyroscope Y in the window |
| `Y_std` | Standard deviation of gyroscope Y |
| `Y_rms` | Root-mean-square of gyroscope Y |

The notebook also computes supplementary in-plane magnitude statistics (`omega_xy_*`), but they are **not** classifier inputs.

**Labels (window level):** a window whose end time is at or before its trial's recorded fall time is **SAFE (0)**. A window that reaches, crosses or follows the fall is **UNSAFE (1)**.

---

## Window Size Selection
A window-size ablation was run with **50% overlap** for every size. Features, labelling rule, trial segmentation, Decision Tree configuration (`max_depth = 5`) and the group-aware split method were identical across experiments, so window size was the only variable.

### Single documented split per window size (Decision Tree, test set)

| Window | Total windows | SAFE | UNSAFE | SAFE % | Usable trials | Train SAFE/UNSAFE | Test SAFE/UNSAFE | Accuracy | Balanced acc. | SAFE recall | UNSAFE recall | Precision (UNSAFE) | F1 (UNSAFE) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.0 s | 18 | 2 | 16 | 11.1% | 16 | 1 / 13 | 1 / 3 | 75.0% | 50.0% | 0.0% | 100.0% | 75.0% | 0.857 |
| 1.0 s | 111 | 41 | 70 | 36.9% | 20 | 33 / 57 | 8 / 13 | 71.4% | 72.1% | 75.0% | 69.2% | 81.8% | 0.750 |
| 0.5 s | 249 | 113 | 136 | 45.4% | 20 | 90 / 111 | 23 / 25 | 77.1% | 76.6% | 65.2% | 88.0% | 73.3% | 0.800 |
| 0.25 s | 533 | 263 | 270 | 49.3% | 20 | 207 / 222 | 56 / 48 | 74.0% | 75.1% | 60.7% | 89.6% | 66.2% | 0.761 |

### Stability across valid group-aware splits
For each window size, up to 25 valid group-aware splits (both classes present in train and test; trial overlap = 0) were evaluated with the same Decision Tree:

| Window | Mean balanced accuracy ± std (25 splits) |
|---|---|
| 3.0 s | 0.485 ± 0.041 |
| 1.0 s | 0.664 ± 0.110 |
| 0.5 s | 0.633 ± 0.101 |
| 0.25 s | 0.632 ± 0.089 |

### Selection
Based on the executed window-size ablation, **1.0 s windows with 50% overlap were selected for the final pipeline for this dataset and experimental setup.**

- **3.0 s** produced too few usable windows (18, with only 2 SAFE; 4 of 20 trials yielded no window). The test set had a single SAFE window, and the Decision Tree predicted UNSAFE for every test window (SAFE recall 0, balanced accuracy 0.5). Its accuracy is therefore not meaningful.
- **1.0 s, 0.5 s and 0.25 s** all made all 20 trials usable and gave valid two-class train and test sets.
- The multi-split balanced accuracies of these three are close relative to their split-to-split variability (std ≈ 0.09–0.11). The notebook's rule treats windows within 0.05 of the best multi-split mean balanced accuracy as tied and picks the **longest** tied window, which keeps the most temporal context per window. Accuracy was not used as the selection criterion.
- **0.5 s** scored higher than 1.0 s on the single documented split (balanced accuracy 0.766 vs 0.721), but this was not reflected in the multi-split mean, so it did not drive the selection.
- **0.25 s** gave substantially more windows but no clear performance advantage.

The multi-split results show considerable variability, so this choice is specific to this small dataset and setup. It should **not** be read as a universal or generalizable optimum.

---

## Final Dataset Configuration

| Item | Value |
|---|---|
| Window size | 1.0 s |
| Overlap | 50% (step = 0.5 s) |
| Usable trials | 20 of 20 |
| Total windows | 111 |
| SAFE / UNSAFE windows | 41 (36.9%) / 70 (63.1%) |
| Split method | `GroupShuffleSplit`, `groups = trial_id`, `test_size = 0.2` |
| Split seed | 42 (first seed giving both classes in train and test; chosen by class presence only, never by model score) |
| Training set | 16 trials, 90 windows (SAFE 33 / UNSAFE 57) |
| Test set | 4 trials, 21 windows (SAFE 8 / UNSAFE 13) |
| Train/test trial overlap | 0 (verified; no train window overlaps a test window in time) |

Because consecutive windows overlap by 50%, windows are **never** split individually. All windows of a trial stay together in train or in test.

---

## Models

### Decision Tree Baseline
- scikit-learn `DecisionTreeClassifier(max_depth=5, random_state=42)`, input: the six features above.
- Fitted tree: depth 5, 10 leaves, 19 nodes.
- This is the **classical baseline**. It is **not** converted to TensorFlow Lite (a scikit-learn tree cannot be directly converted).

### Neural Network
A separate compact TensorFlow/Keras model, trained on the same six features, is the model used for TFLite conversion.

```
Input (6 features, standardized)
  → Dense(16, ReLU)
  → Dense(8, ReLU)
  → Dense(1, Sigmoid)
```

- Optimizer: Adam; loss: binary cross-entropy.
- 50 epochs, batch size 16.
- Features are standardized with a `StandardScaler` **fitted on the training set only**; its parameters are saved to `feature_scaler.json` and reused at TFLite inference time.
- The test set is passed as `validation_data` only to plot the learning curve. There is no early stopping, checkpointing or hyperparameter selection based on it.
- Final training accuracy: 0.7667.

---

## TensorFlow Lite Conversion
The trained **Keras neural network** is converted with `tf.lite.TFLiteConverter.from_keras_model`.

### Float32 Model
- Direct conversion with no quantization (`model_float32.tflite`, 3,292 bytes ≈ 3.21 KB).

### INT8 Quantized Model
- Post-training **full-integer quantization** of the same Keras model (`model_int8.tflite`, 3,432 bytes ≈ 3.35 KB).
- Converter settings: `Optimize.DEFAULT`, `OpsSet.TFLITE_BUILTINS_INT8`, `int8` inference input and output types.
- **Representative dataset:** the 90 **training** samples (scaled features). Test data are never used for calibration.
- Measured tensor details: input `int8` (scale 0.03198, zero-point −27); output `int8` (scale 0.00390625, zero-point −128).
- At this model size, the INT8 file is slightly **larger** than the Float32 file (−4.25% "reduction"), since quantization metadata can outweigh the weight savings for a network of a few hundred parameters.

### TFLite Interpreter Inference
Both `.tflite` models are loaded with `tf.lite.Interpreter` and run on the held-out test set, one sample at a time, inside the notebook. Inputs are standardized with the saved training-set scaler and (for INT8) quantized using the interpreter's own scale and zero-point. Outputs are dequantized and thresholded at 0.5. Edge execution is **simulated** in the notebook; no on-device deployment is performed.

---

## Evaluation
All metrics below are computed on the held-out test set (21 windows: 8 SAFE, 13 UNSAFE). UNSAFE is the positive class for precision, recall and F1. SAFE recall is the true-negative rate.

**Balanced accuracy** is the mean of SAFE recall and UNSAFE recall. It is more informative than accuracy when class proportions differ, because a model that predicts one class for everything scores only 0.5.

| Model | Accuracy | Balanced acc. | Precision | UNSAFE recall | SAFE recall | F1 | Confusion (TN / FP / FN / TP) |
|---|---:|---:|---:|---:|---:|---:|---|
| Decision Tree | 0.7143 | 0.7212 | 0.8182 | 0.6923 | 0.7500 | 0.7500 | 6 / 2 / 4 / 9 |
| Keras (float) | 0.7619 | 0.7356 | 0.7857 | 0.8462 | 0.6250 | 0.8148 | 5 / 3 / 2 / 11 |
| TFLite Float32 | 0.7619 | 0.7356 | 0.7857 | 0.8462 | 0.6250 | 0.8148 | 5 / 3 / 2 / 11 |
| TFLite INT8 | 0.7619 | 0.7356 | 0.7857 | 0.8462 | 0.6250 | 0.8148 | 5 / 3 / 2 / 11 |

**Quantization fidelity:** Float32 TFLite and INT8 TFLite predictions agree with the Keras model on 100% of test windows (max |probability difference| for INT8 vs Keras = 0.0068).

**Notebook-based latency** (single-sample inference, 200 timed runs after 20 warm-up runs, in the notebook's CPU environment):

| Model | Size (KB) | Mean (ms) | Median (ms) | Std (ms) |
|---|---:|---:|---:|---:|
| Float32 TFLite | 3.21 | 0.0078 | 0.0047 | 0.0306 |
| INT8 TFLite | 3.35 | 0.0044 | 0.0029 | 0.0043 |

At this model size the measured time is dominated by Python/interpreter call overhead on a desktop/cloud CPU. Values vary between runs and do not predict microcontroller performance.

---

## Experimental Scope and Limitations
- **All 20 physical trials contain a failure event.** There are no independently recorded SAFE trials. SAFE and UNSAFE are assigned at the **window level** around each recorded fall, so the task is closer to *pre-failure vs failure-window* classification than to *SAFE-trial vs UNSAFE-trial* classification.
- **The dataset is small** (20 trials from one continuous recording session). Drift, mounting effects or handling artefacts of that session affect every trial.
- **Fall times were recorded with a manual stopwatch** and matched to the nearest gyroscope timestamp, which introduces an unquantified timing offset in the labels.
- **Overlapping windows are not independent experiments.** Windows from the same trial are strongly correlated, so the number of windows overstates the amount of independent evidence. Trial-aware splitting is therefore essential. Shorter windows create more windows, not more independent trials.
- **The test set is small** (4 trials, 21 windows at 1.0 s), and the multi-split results show large variability. Single-split metrics should be read as indicative only.
- No samples were duplicated or synthesised, and no sensor values or labels were altered.
- The results are a **proof-of-concept** and should not be interpreted as production-level or real-world generalization.

---

## Repository Structure (Branch: `Lab-1`)

```
├── 2548560_Tejas_R_M_TinyML_LAB01_Real_Time_Rotational_Dynamics_Edge_AI_Payload_Safety_System.ipynb
├── Main_Raw Data.csv
├── model_float32.tflite
├── model_int8.tflite
├── Workflow.png
└── README.md
```

---

## Requirements & Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## How to Run
1. Clone or checkout the `Lab-1` branch:
```bash
   git clone -b Lab-1 https://github.com/Tejas2913/TinyML-Lab-2548560.git
```
2. Open the notebook:
```bash
   jupyter notebook 2548560_Tejas_R_M_TinyML_LAB01_Real_Time_Rotational_Dynamics_Edge_AI_Payload_Safety_System.ipynb
```
3. Place `Main_Raw Data.csv` where the notebook expects it. The configuration cell reads `dataset/raw/Main_Raw Data.csv` relative to the working directory.
4. Run the cells sequentially. The window-size ablation selects the window used by the rest of the pipeline, and the notebook writes figures, processed features, reports and the `.tflite` models to an `outputs/` folder. Exact latency values will differ between machines.
