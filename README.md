# **CAMELS STREAMFLOW PREDICTION PIPELINE**

---

## **1. PROJECT OVERVIEW**

This project implements a Deep Learning pipeline for hydrological streamflow prediction using the CAMELS (Catchment Attributes and Meteorology for Large-sample Studies) dataset.

The pipeline processes meteorological forcing data (precipitation, temperature, etc.) and static catchment attributes to predict future streamflow (discharge).

---

## **2. DATASET**

**Source:** [https://zenodo.org/records/15529996](https://zenodo.org/records/15529996)
The code expects the dataset to be located at:

```
../basin_dataset_public
```

**Expected directory structure:**

```
/basin_dataset_public
    /basin_mean_forcing/nldas     (Forcing data)
    /usgs_streamflow              (Target data)
    /basin_size_errors...txt      (Bad basins list)
    ...                           (Static attribute files)
```

---

## **3. TASKS**

The pipeline supports two distinct prediction tasks:

### **[Task 1] Early Warning System (Single-Step)**

* **Goal:** Predict streamflow at time (t + 2 days).
* **Model:** Standard LSTM.
* **Input:** Past dynamic features + Static attributes.

### **[Task 2] Operational Planning (Multi-Step Sequence)**

* **Goal:** Predict the streamflow sequence for the next 5 days (t+1 to t+5).
* **Model:** Encoder–Decoder LSTM (Seq2Seq) with Cross-Attention.
* **Input:** Past dynamic features, future known forcing (weather forecasts), and static attributes.

---

## **4. PIPELINE STAGES**

1. **Data Loading:** Scans directories, filters “bad” basins, loads raw text files.
2. **Cleaning:** Removes physical outliers (negative rain, unrealistic temperatures).
3. **Feature Engineering:** Adds rolling averages (3d, 7d) and lag features.
4. **Preprocessing:**

   * Handles missing data (interpolation).
   * Normalizes data (StandardScaler) using training statistics.
   * Generates sine/cosine features for dates.
5. **Modeling:** PyTorch implementations of LSTM and Seq2Seq Attention.
6. **Evaluation:** Nash–Sutcliffe Efficiency (NSE) score.

---

## **5. FILE STRUCTURE**

```
config.py              : Configuration constants (paths, dates, hyperparameters)
loader.py              : Handles raw file I/O and directory scanning
preprocessing.py       : Data cleaning, normalization, and sequence creation
feature_engineering.py : Generates rolling stats and lag features
model.py               : PyTorch model architectures (LSTM, Attention)
main.py                : CLI entry point for training and testing
```

---

## **6. REQUIREMENTS**

* Python 3.x
* PyTorch
* NumPy
* Pandas
* Tqdm

---

## **7. HOW TO RUN**

The pipeline is executed via `main.py` using command-line arguments.

**Arguments:**

```
--task [1, 2, all]     : Which task to run
--num_basins [int]     : Number of basins to load (0 = All basins)
                         (Use a low number like 5 for debugging)
--epochs [int]         : Number of training epochs
--batch_size [int]     : Batch size
--gpu                  : Enable GPU training
```

**Examples:**

```bash
# 1. Run Task 1 (t+2 prediction) on 10 basins for quick test
python main.py --task 1 --num_basins 10 --epochs 5
```

```bash
# 2. Run Task 2 (Sequence t+1..t+5) on ALL basins using GPU
python main.py --task 2 --num_basins 0 --epochs 20 --gpu
```

```bash
# 3. Run Both Tasks sequentially
python main.py --task all --epochs 10
```

---

## **8. RESULTS**

Results (metrics and model checkpoints) are saved in the `results/` directory:

```
results/task1/metrics.json
results/task2/metrics.json
```

---
