# Speech Emotion Recognition

## Overview

This project implements a Speech Emotion Recognition (SER) system using machine learning and deep learning techniques. The system aims to recognize human emotions such as happiness, sadness, anger, fear, calm, and neutral from speech audio by analyzing features like tone and pitch.

## Features

- Audio preprocessing and feature extraction using librosa (MFCC, chroma, mel spectrogram, etc.)
- Deep learning-based emotion classification model (e.g., CNN, LSTM)
- Supports multiple emotion categories
- Evaluation via accuracy, confusion matrix, and classification reports
- User-friendly frontend interface for live or recorded speech emotion prediction

## Frontend Screenshot

Below is a screenshot of the frontend interface that allows users to upload or record speech and view predicted emotions in real-time:

![Frontend Screenshot](images/frontend_screenshot.png)

*Replace `images/frontend_screenshot.png` with the relative path of your screenshot file in the repo.*

## Tech Stack

- Python 3.x
- TensorFlow / Keras (for deep learning models)
- librosa (for audio feature extraction)
- scikit-learn (for data preprocessing and evaluation)
- Flask or Streamlit (for frontend web application)
- NumPy, pandas (for data manipulation)
- Matplotlib or seaborn (for visualization)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/kani2006/Speech-Emotion-Recognition.git
   cd Speech-Emotion-Recognition
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## How to Run

### Step 1: Prepare Dataset
- Download a speech emotion dataset (e.g., RAVDESS, CREMA-D, TESS) and place it in the `data` directory.
- Make sure the dataset is organized with labeled emotion categories.

### Step 2: Extract Features
Run the feature extraction script to process audio files and extract features:
```
python preprocess.py --dataset_path ./data
```

### Step 3: Train the Model
Train the SER model on the preprocessed features:
```
python train.py --epochs 50 --batch_size 32
```

### Step 4: Evaluate the Model
Evaluate performance on the test set:
```
python evaluate.py --model_path ./saved_model
```

### Step 5: Run the Frontend Application
Start the web application interface for real-time emotion prediction:
```
python app.py
```
- Use the frontend to record or upload audio and see emotion predictions instantly.

## Dataset

Public datasets such as RAVDESS, CREMA-D, and TESS are supported for training and evaluation.

## Results
<img width="1846" height="991" alt="Screenshot from 2025-10-01 14-35-16" src="https://github.com/user-attachments/assets/acec90da-3149-4fa0-8033-36ae866a2831" />


## Contributing

Contributions are welcome. Feel free to fork, raise issues, or submit pull requests.

## License

This project is licensed under the MIT License.

## Acknowledgements

Thanks to data providers and open-source libraries that enabled this project.

---

Happy coding with Speech Emotion Recognition!
