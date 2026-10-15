# Breast Cancer Detection with a Neural Network

A binary classifier built with TensorFlow/Keras that predicts whether a breast tumour is malignant or benign from 30 cell-nucleus measurements. It uses the Breast Cancer Wisconsin dataset bundled with scikit-learn, so no download is needed.

## Features

- Stratified 80/20 train/test split and feature standardization
- Neural network: Dense(32) → Dropout(0.2) → Dense(16) → Dense(1, sigmoid)
- Training and validation accuracy/loss curves
- Accuracy, classification report and confusion matrix
- Single-sample prediction with probability

## Project Structure

```
breast-cancer-detection/
├── breast_cancer_detection.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/breast-cancer-detection.git
cd breast-cancer-detection
pip install -r requirements.txt
```

## Usage

```bash
python breast_cancer_detection.py
```

Plots are saved as `training_history.png` and `confusion_matrix.png`.

## Notes

- In this dataset class `0` is malignant and `1` is benign.
- Fixed random seeds make runs similar, but TensorFlow results can still vary slightly between machines.

## Disclaimer

For educational purposes only. This is not a medical device and must not be used for diagnosis or treatment decisions.

## License

MIT
