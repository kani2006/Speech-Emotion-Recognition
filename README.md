# Speech Emotion Recognition

A comprehensive, deep learning-based project and web demo for recognizing emotions from speech using audio inputs. This project unifies six major public emotion datasets and delivers robust neural models for emotion classification, with an interactive, empathetic chatbot frontend.

---

## Features

- Supports six major datasets: RAVDESS, TESS, CREMA-D, EMO-DB, IEMOCAP, and SAVEE.
- Unifies and maps emotion labels from various datasets to a common taxonomy for joint or cross-dataset training.
- Deep neural network model with batch normalization, dropout, LeakyReLU activations, and Adam optimizer.
- Automatic feature extraction (MFCCs) and pre-processing from audio files.
- Robust label encoding, data balancing, and performance visualization.
- Saves models, encoders, and training history for reproducible experiments.
- Includes a frontend for interactive chat, accepting audio (recording/upload) or text input, and generating empathetic AI chatbot replies with emotion feedback.
- Web UI shows analyzed emotions, confidence scores, and friendly feedback (including emoji/avatars).

---

## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/speech-emotion-recognition.git
   cd speech-emotion-recognition
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Prepare dataset folders:**
   - Download each dataset (RAVDESS, TESS, CREMA-D, EMO-DB, IEMOCAP, SAVEE).
   - Update the `DATASETPATHS` in the main script to set local dataset paths.

4. **Configuration:**
   - Adjust main parameters, model save paths, and optionally set up Gemini or OpenAI API keys for advanced chatbot features in the UI.

---

## Usage

### Model Training

1. Run the main training script:
   ```
   python train.py
   ```
   - Training runs for 100 epochs with default parameters.
   - Final model saved to `/models/enhancedemotionmodel.h5`.
   - Label encoder and training history saved for later use.

2. Training and validation progress is plotted and saved automatically.

### Running the Web Application

1. Run the Flask backend (after training and saving the model):
   ```
   python app.py
   ```
2. Open the frontend in your browser (usually at `http://localhost:5000`).

3. Interact by sending audio or text messages; the assistant detects the emotional state and replies empathetically.

---

## Example

- Record or upload an audio sample, or type a message such as `I'm feeling stressed…`
- The assistant analyzes the input, displays detected emotion (e.g., "sad", "angry") with confidence, and gives a warm, supportive response.

---

## Project Structure

| File/Folder        | Purpose                                 |
|--------------------|-----------------------------------------|
| `train_model.py`         | Model training and dataset integration  |
| `run_ai.py`         | Command line output  |
| `models/`          | Saved models and encoders               |
| `template`        | Web UI code                             |
| `app.py`        | Backend Flask API for the web frontend                              |

---

## Datasets

- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song
- **TESS**: Toronto Emotional Speech Set
- **CREMA-D**: Crowd-sourced Emotional Multimodal Actor Dataset
- **EMO-DB**: Berlin Emotional Speech Database
- **IEMOCAP**: Interactive Emotional Dyadic Motion Capture Database
- **SAVEE**: Surrey Audio-Visual Expressed Emotion Database

Follow each dataset’s official source for download and usage terms.

---
## Result
<img width="1846" height="991" alt="Screenshot from 2025-10-01 14-35-16" src="https://github.com/user-attachments/assets/ba80178c-2cd2-4580-bafa-21e54ecc6b75" />

---

## Acknowledgments

Thanks to contributors, academic sources, and open dataset providers that made robust emotion recognition research possible[file:1].
