# 🤟 Sign Language Recognition using Deep Learning
*(Beginner-friendly version — Sign Language MNIST dataset)*

A CNN-based project that recognizes ASL alphabet letters, with a Streamlit
app supporting both:
- 📁 Image upload recognition
- 📷 Real-time live webcam recognition

This version uses a small, easy-to-download dataset — good for an
internship / early-year project where you want something you can train
and demo quickly without a huge download or a GPU.

---

## 1. Dataset

**Sign Language MNIST** — Kaggle
https://www.kaggle.com/datasets/datamunge/sign-language-mnist

- Only **2 CSV files**, ~63 MB total (`sign_mnist_train.csv`, `sign_mnist_test.csv`)
- 28x28 grayscale images, already flattened into pixel columns — just like
  the classic MNIST digits dataset, but for hand signs
- 24 classes: A–Z, excluding **J** and **Z** (those need hand motion, not
  a static pose, so this dataset leaves them out)
- ~27,000 training rows, ~7,000 test rows

### How to download
1. Create a free Kaggle account.
2. Go to the dataset page above → click **Download** (no API setup needed
   for a manual download — just unzip it).
3. Put both CSVs in a `dataset/` folder next to `train_model.py`:
   ```
   dataset/
       sign_mnist_train.csv
       sign_mnist_test.csv
   ```

*(If you'd rather use the Kaggle API instead of a manual download:
`pip install kaggle` then `kaggle datasets download -d datamunge/sign-language-mnist`.)*

---

## 2. Project structure

```
sign_language_project/
├── train_model.py           # trains the CNN and saves the model
├── app.py                   # Streamlit app (upload + live webcam)
├── requirements.txt
├── utils/
│   └── predict.py           # shared preprocessing/prediction helpers
└── model/                   # created after training
    ├── sign_language_model.h5
    ├── class_indices.json
    └── training_curves.png
```

---

## 3. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Train the model

```bash
python train_model.py --data_dir dataset --epochs 15
```

This will:
- Load both CSVs and reshape each row back into a 28x28 grayscale image
- Train a small 3-block CNN (Conv2D + MaxPool, doubling filters each block)
- Use the provided test CSV as validation during training
- Save the best model to `model/sign_language_model.h5`
- Save the label map to `model/class_indices.json`
- Save accuracy/loss curves to `model/training_curves.png` (useful for your report)

On a normal laptop CPU this finishes in a few minutes — no GPU required.

---

## 5. Run the app

```bash
streamlit run app.py
```

- **📁 Image Upload tab:** upload a hand-sign photo → get the predicted
  letter, confidence score, and top-5 probabilities.
- **📷 Live Webcam tab:** grants camera access via `streamlit-webrtc`,
  overlays the live predicted letter + confidence box on the video feed.

**Note:** since the underlying dataset only has static-pose letters, the
model can't recognize **J** or **Z** (both require a hand movement, not
just a pose).

---

## 6. What to say in your report/demo

- **Problem:** Recognize static ASL alphabet signs from an image.
- **Approach:** CNN trained on 28x28 grayscale hand-sign images.
- **Why this dataset:** small size (~63 MB), quick to train, no GPU needed —
  good fit for a short internship timeline.
- **Extensions (mention as future work):** move to a larger RGB dataset
  (e.g., Kaggle's ASL Alphabet, ~87k images) for higher real-world webcam
  accuracy, or add motion-based signs (J, Z) using frame sequences.

---

## 7. Tech stack

| Component        | Tool                          |
|-------------------|--------------------------------|
| Model             | CNN (TensorFlow/Keras)          |
| Image processing  | OpenCV, Pillow                  |
| Live video        | streamlit-webrtc                |
| UI                | Streamlit                       |
| Dataset           | Kaggle Sign Language MNIST       |
