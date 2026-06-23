# Cat vs Dog Image Classifier

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images as either cats or dogs. The trained model is exported as a TFLite file for lightweight deployment.

---

## Overview

This project trains a binary image classifier on the [Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset) from Kaggle. It is designed to run in Google Colab and produces a `.tflite` model file suitable for mobile or edge deployment.

---

## Requirements

- Python 3.x
- TensorFlow
- Kaggle Hub (`kagglehub`)
- NumPy
- Matplotlib
- Google Colab (for dataset download and file export)

Install dependencies:

```bash
pip install tensorflow kagglehub numpy matplotlib
```

---

## Project Structure

```
Cat___Dog.ipynb       # Main Jupyter notebook
cat_dog_model.tflite  # Exported TFLite model (generated after training)
```

---

## Pipeline

### Step 1 — Download Dataset
Downloads the dataset from Kaggle using `kagglehub`:
```python
path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
```
The notebook automatically detects whether images are nested inside a `PetImages/` folder.

### Step 2 — Preprocess Images
Uses `ImageDataGenerator` to:
- Rescale pixel values to `[0, 1]`
- Auto-split data: **80% training / 20% validation**
- Resize all images to **150×150 pixels**
- Load in batches of **32**

### Step 3 — Build CNN Model
A sequential CNN with the following architecture:

| Layer | Details |
|---|---|
| Conv2D + MaxPooling | 32 filters, 3×3 kernel, ReLU |
| Conv2D + MaxPooling | 64 filters, 3×3 kernel, ReLU |
| Conv2D + MaxPooling | 128 filters, 3×3 kernel, ReLU |
| Flatten | — |
| Dense | 512 units, ReLU |
| Dense (output) | 1 unit, Sigmoid (binary classification) |

### Step 4 — Compile & Train
- **Loss:** Binary cross-entropy
- **Optimizer:** Adam
- **Metric:** Accuracy
- **Epochs:** 10

### Step 5 — Export to TFLite
The trained Keras model is converted and saved as `cat_dog_model.tflite` for deployment on mobile or embedded devices.

### Step 6 — Visualize Results
Training and validation accuracy/loss curves are plotted using Matplotlib to evaluate model performance over epochs.

---

## Output

- `cat_dog_model.tflite` — Lightweight model for inference on mobile/edge devices
- Training/validation accuracy and loss plots

---

## Usage

1. Open `Cat___Dog.ipynb` in Google Colab.
2. Run all cells in order.
3. The `.tflite` model will be automatically downloaded to your machine after training.

---

## Dataset

**Source:** [Kaggle — Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset) by Bhavikh Jikadara

**Classes:** `Cat`, `Dog` (binary classification)
