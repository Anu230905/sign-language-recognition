"""
train_model.py
---------------
Trains a simple CNN on the Kaggle "Sign Language MNIST" dataset.
This is a LIGHTWEIGHT dataset (just 2 CSV files, ~63 MB total) -
great for a college internship project, trains in a couple of
minutes even without a GPU.

Dataset link: https://www.kaggle.com/datasets/datamunge/sign-language-mnist

Expected files (download and place in a "dataset" folder):
    dataset/sign_mnist_train.csv
    dataset/sign_mnist_test.csv

Each CSV row = one image:
    column 0       -> label (0-25, skips 9=J and 25=Z since those need motion)
    columns 1-784  -> pixel values (28x28 image, flattened, grayscale)

Usage:
    python train_model.py --data_dir dataset --epochs 15
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 28

# The dataset skips letters J (9) and Z (25) because they require hand motion,
# not a static pose. This map converts the CSV's label number -> actual letter.
LABEL_TO_LETTER = {i: chr(65 + i) for i in range(26) if i not in (9, 25)}


def load_csv_dataset(csv_path):
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values.astype("float32") / 255.0
    images = pixels.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return images, labels


def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(args):
    train_csv = os.path.join(args.data_dir, "sign_mnist_train.csv")
    test_csv = os.path.join(args.data_dir, "sign_mnist_test.csv")

    print("Loading data...")
    X_train, y_train_raw = load_csv_dataset(train_csv)
    X_test, y_test_raw = load_csv_dataset(test_csv)

    # Build a clean 0..N-1 label index (since original labels skip 9 and 25)
    unique_labels = sorted(set(y_train_raw.tolist()))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_letter = {i: LABEL_TO_LETTER[label] for label, i in label_to_idx.items()}

    y_train_idx = np.array([label_to_idx[l] for l in y_train_raw])
    y_test_idx = np.array([label_to_idx[l] for l in y_test_raw])

    num_classes = len(unique_labels)
    y_train = to_categorical(y_train_idx, num_classes)
    y_test = to_categorical(y_test_idx, num_classes)

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Number of classes: {num_classes}")

    # Save the label map so the Streamlit app can decode predictions
    os.makedirs("model", exist_ok=True)
    with open("model/class_indices.json", "w") as f:
        json.dump(idx_to_letter, f, indent=2)

    # Light augmentation directly with tf.data (dataset is small, so this is fast)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    datagen.fit(X_train)

    model = build_model(num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "model/sign_language_model.h5",
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal test accuracy: {test_acc*100:.2f}%")

    # Plot accuracy/loss curves for your project report
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig("model/training_curves.png")
    print("\nSaved model to model/sign_language_model.h5")
    print("Saved class map to model/class_indices.json")
    print("Saved training curves to model/training_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Folder containing sign_mnist_train.csv and sign_mnist_test.csv",
    )
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    main(args)
