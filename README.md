# 🧠 Shallow & Improved CNN for COVID-19 Detection from Chest X-Rays

This project is a replication of the paper:  
**"Shallow CNN for COVID-19 Outbreak Screening Using Chest X-rays"**,  
and additionally includes an improved version of the model for enhanced performance.

---

## 📌 Objective

To automatically classify chest X-ray (CXR) images into three categories:
- **COVID-19**
- **PNEUMONIA**
- **NORMAL**

> Note: While the original paper focused on binary classification (COVID vs non-COVID), this implementation extends it to three classes for more practical usage.

---

## 🏗️ Architecture

### 1. ShallowCNN (from the paper)
- 1 × Conv2D layer (10 filters, 2×2)
- MaxPooling layer (2×2)
- Fully connected layer (256 units, ReLU)
- Output layer (Softmax)

> ⚠️ This architecture is included in `main.py` but **commented out** by default.

### 2. ImprovedCNN (active)
- 3 × Conv2D layers (increasing filters)
- MaxPooling
- Dropout for regularization
- Fully connected layers
- Output layer for 3-class classification

---

## 🗂️ Dataset

Used the publicly available **chest_xray** dataset from Kaggle and the COVID samples from [ieee8023 GitHub](https://github.com/ieee8023/covid-chestxray-dataset).

- Total: ~6400 images
- Train / Validation / Test split:
  - `train/` — 5666 images
  - `val/` — 72 images
  - `test/` — 681 images

> Dataset structure follows the format expected by PyTorch’s `ImageFolder`.

---

## 🚀 How to Run

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your dataset in the `chest_xray/` folder with the following structure:
   ```
   chest_xray/
       ├── train/
       ├── val/
       └── test/
   ```

3. Run the training script:
   ```bash
   python main.py
   ```

4. Output:
   - Console will display epoch-wise accuracy, loss
   - Generated plots:
     - `loss_plot.png`
     - `val_accuracy_plot.png`
     - `confusion_matrix.png`
     - `sample_predictions.png`

---

## 📈 Results

| Model        | Test Accuracy | Weighted F1-score |
|--------------|---------------|-------------------|
| ShallowCNN   | 75.77%        | 0.72              |
| ImprovedCNN  | 78.56%        | 0.76              |

---

## 🧑‍💻 Author

- **Ostap Spas**
- **Ivan Hrechkovskiy**

---

## 📄 License

This project is for academic and research purposes only.