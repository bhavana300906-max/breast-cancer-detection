"""Breast cancer detection with a Keras neural network (scikit-learn dataset)."""

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

SEED = 42
tf.keras.utils.set_random_seed(SEED)


def build_model(n_features):
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_history(history):
    plt.figure(figsize=(12, 4))
    for i, (metric, title) in enumerate([("accuracy", "Model Accuracy"), ("loss", "Model Loss")], 1):
        plt.subplot(1, 2, i)
        plt.plot(history.history[metric], label=f"Training {metric.title()}")
        plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric.title()}")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Breast Cancer Detection - Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    print("Dataset Shape:", X.shape)
    print("\nFirst 5 Rows:")
    print(X.head())
    print("\nTarget Names:", data.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Training Data Shape:", X_train.shape)
    print("Testing Data Shape:", X_test.shape)

    model = build_model(X_train.shape[1])
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=50, batch_size=16, validation_split=0.2, verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Test Loss:", round(loss, 4))
    print("Test Accuracy:", round(accuracy * 100, 2), "%")

    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob >= 0.5).astype(int).flatten()
    print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    plot_history(history)

    # Predict the first test sample (class 0 = malignant, 1 = benign)
    probability = float(model.predict(X_test[0].reshape(1, -1), verbose=0)[0][0])
    result = "Benign" if probability >= 0.5 else "Malignant"
    print("Predicted Result:", result)
    print("Probability of Benign:", round(probability * 100, 2), "%")

    plot_confusion_matrix(cm, data.target_names)


if __name__ == "__main__":
    main()
