import os
import re
import librosa
import numpy as np
import soundfile as sf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import pickle
import json
from datetime import datetime

matplotlib.use("Agg")

# Dataset paths
DATASET_PATHS = {
    'RAVDESS': '/media/victus/ee48b6cb-9d85-46cc-9dff-c15451b93cf0/SER_Project/dataset/RAVDESS',
    'TESS': '/media/victus/ee48b6cb-9d85-46cc-9dff-c15451b93cf0/SER_Project/dataset/tess/toronto-emotional-speech-set-tess',
    'CREMA_D': '/media/victus/ee48b6cb-9d85-46cc-9dff-c15451b93cf0/SER_Project/dataset/CREMA-D',
    'EMO_DB': '/media/victus/ee48b6cb-9d85-46cc-9dff-c15451b93cf0/SER_Project/dataset/EMO-DB',
    'IEMOCAP': '/media/victus/ee48b6cb-9d85-46cc-9dff-c15451b93cf0/SER_Project/dataset/IEMOCAP',
    'SAVEE': '/media/victus/ee48b6cb-9d85-46cc-9dff-c15451b93cf0/SER_Project/dataset/SAVEE'
}

MODEL_SAVE_PATH = 'models/'
MODEL_NAME = 'enhanced_emotion_model.h5'
ENCODER_NAME = 'label_encoder.pkl'
TRAINING_HISTORY_NAME = 'training_history.json'

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Emotion mappings for each dataset
EMOTION_MAPPINGS = {
    'RAVDESS': {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    },
    'TESS': {
        'angry': 'angry', 'disgust': 'disgust', 'fear': 'fearful',
        'happy': 'happy', 'neutral': 'neutral', 'ps': 'surprised', 'sad': 'sad'
    },
    'CREMA_D': {
        'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful', 'HAP': 'happy',
        'NEU': 'neutral', 'SAD': 'sad'
    },
    'EMO_DB': {
        'W': 'angry', 'L': 'boredom', 'E': 'disgust', 'A': 'fearful',
        'F': 'happy', 'T': 'sad', 'N': 'neutral'
    },
    'IEMOCAP': {
        'ang': 'angry', 'hap': 'happy', 'exc': 'excited', 'sad': 'sad',
        'fru': 'frustrated', 'fea': 'fearful', 'sur': 'surprised', 'neu': 'neutral'
    },
    'SAVEE': {
        'a': 'angry', 'd': 'disgust', 'f': 'fearful', 'h': 'happy',
        'n': 'neutral', 'sa': 'sad', 'su': 'surprised'
    }
}

def extract_features(file_path, max_pad_len=174):
    """Extract MFCC features from audio file"""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def find_all_audio_files(path, extensions=['.wav', '.mp3', '.flac', '.m4a']):
    """Recursively find all audio files in a directory"""
    audio_files = []
    if not os.path.exists(path):
        return audio_files
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    
    return audio_files

def explore_dataset_structure(dataset_name, path):
    """Explore and display dataset structure"""
    print(f"🔍 Exploring {dataset_name} structure at: {path}")
    
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return 0
    
    audio_files = find_all_audio_files(path)
    print(f"Found {len(audio_files)} audio files in {dataset_name}")
    
    # Show directory structure (first few levels)
    if audio_files:
        print("Directory structure sample:")
        shown_dirs = set()
        for file_path in audio_files[:10]:  # Show first 10 files' directories
            rel_path = os.path.relpath(os.path.dirname(file_path), path)
            if rel_path not in shown_dirs and rel_path != '.':
                print(f"   {rel_path}")
                shown_dirs.add(rel_path)
        if len(audio_files) > 10:
            print(f"   ... and more directories")
    
    return len(audio_files)

def parse_ravdess_filename(filename):
    """Parse RAVDESS filename format: Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav"""
    try:
        parts = filename.replace('.wav', '').split('-')
        if len(parts) >= 3:
            emotion_code = parts[2]
            return EMOTION_MAPPINGS['RAVDESS'].get(emotion_code)
    except Exception as e:
        print(f"Error parsing RAVDESS filename {filename}: {e}")
    return None

def parse_crema_d_filename(filename):
    """Parse CREMA-D filename format: ActorID_Sentence_Emotion_Intensity.wav"""
    try:
        parts = filename.replace('.wav', '').split('_')
        if len(parts) >= 3:
            emotion_code = parts[2]
            return EMOTION_MAPPINGS['CREMA_D'].get(emotion_code)
    except Exception as e:
        print(f"Error parsing CREMA-D filename {filename}: {e}")
    return None

def parse_emo_db_filename(filename):
    """Parse EMO-DB filename format: [Speaker][Text][Emotion][Version].wav"""
    try:
        # EMO-DB format: e.g., "03a01Wa.wav" - emotion is usually the second to last character
        base_name = filename.replace('.wav', '')
        if len(base_name) >= 6:
            emotion_code = base_name[-2]  # Second to last character
            return EMOTION_MAPPINGS['EMO_DB'].get(emotion_code)
    except Exception as e:
        print(f"Error parsing EMO-DB filename {filename}: {e}")
    return None

def parse_iemocap_filename(filename):
    """Parse IEMOCAP filename - emotion usually in filename or folder"""
    try:
        filename_lower = filename.lower()
        for emotion_code, emotion in EMOTION_MAPPINGS['IEMOCAP'].items():
            if emotion_code in filename_lower:
                return emotion
    except Exception as e:
        print(f"Error parsing IEMOCAP filename {filename}: {e}")
    return None

def parse_savee_filename(filename):
    """Parse SAVEE filename format: [Speaker]_[Emotion][Number].wav"""
    try:
        base_name = filename.replace('.wav', '').lower()
        # SAVEE format: e.g., "DC_a01.wav", "JE_h01.wav"
        parts = base_name.split('_')
        if len(parts) >= 2:
            emotion_part = parts[1]
            # Extract emotion letter(s)
            if emotion_part.startswith('sa'):
                return EMOTION_MAPPINGS['SAVEE'].get('sa')
            elif emotion_part.startswith('su'):
                return EMOTION_MAPPINGS['SAVEE'].get('su')
            else:
                emotion_code = emotion_part[0]
                return EMOTION_MAPPINGS['SAVEE'].get(emotion_code)
    except Exception as e:
        print(f"Error parsing SAVEE filename {filename}: {e}")
    return None

def parse_tess_filename(filename, folder_path):
    """Parse TESS filename and folder structure"""
    try:
        filename_lower = filename.lower()
        folder_name = os.path.basename(folder_path).lower()
        
        # Check filename first
        for emotion_code, emotion in EMOTION_MAPPINGS['TESS'].items():
            if emotion_code in filename_lower:
                return emotion
        
        # Check folder name
        for emotion_code, emotion in EMOTION_MAPPINGS['TESS'].items():
            if emotion_code in folder_name:
                return emotion
    except Exception as e:
        print(f"Error parsing TESS filename {filename}: {e}")
    return None

def load_dataset(dataset_name, path):
    """Load a specific dataset"""
    print(f"\n📊 Loading {dataset_name} dataset...")
    features = []
    labels = []
    file_count = 0
    error_count = 0
    
    if not os.path.exists(path):
        print(f"{dataset_name} path not found: {path}")
        return [], []
    
    # Explore structure first
    total_files = explore_dataset_structure(dataset_name, path)
    if total_files == 0:
        print(f"No audio files found in {dataset_name}")
        return [], []
    
    print(f"🎵 Processing {dataset_name} files...")
    
    # Get all audio files
    audio_files = find_all_audio_files(path)
    
    for file_path in audio_files:
        filename = os.path.basename(file_path)
        folder_path = os.path.dirname(file_path)
        
        # Parse emotion based on dataset
        emotion = None
        if dataset_name == 'RAVDESS':
            emotion = parse_ravdess_filename(filename)
        elif dataset_name == 'TESS':
            emotion = parse_tess_filename(filename, folder_path)
        elif dataset_name == 'CREMA_D':
            emotion = parse_crema_d_filename(filename)
        elif dataset_name == 'EMO_DB':
            emotion = parse_emo_db_filename(filename)
        elif dataset_name == 'IEMOCAP':
            emotion = parse_iemocap_filename(filename)
        elif dataset_name == 'SAVEE':
            emotion = parse_savee_filename(filename)
        
        if emotion:
            # Extract features
            mfccs = extract_features(file_path)
            if mfccs is not None:
                features.append(mfccs)
                labels.append(emotion)
                file_count += 1
                
                if file_count % 100 == 0:
                    print(f"   Processed {file_count}/{total_files} {dataset_name} files...")
            else:
                error_count += 1
        else:
            if error_count < 10:  # Only show first 10 errors to avoid spam
                print(f"   ❓ Could not determine emotion for: {filename}")
            error_count += 1
    
    print(f" {dataset_name}: Successfully loaded {file_count} files")
    if error_count > 0:
        print(f"{dataset_name}: {error_count} files had errors or unknown emotions")
    
    return features, labels

def create_combined_dataset():
    """Create combined dataset from all available datasets"""
    print("🚀 CREATING COMPREHENSIVE COMBINED DATASET")
    print("="*80)
    
    all_features = []
    all_labels = []
    dataset_stats = {}
    
    # Check which datasets are available
    available_datasets = []
    for dataset_name, path in DATASET_PATHS.items():
        if os.path.exists(path):
            available_datasets.append(dataset_name)
            print(f" {dataset_name}: Found at {path}")
        else:
            print(f" {dataset_name}: Not found at {path}")
    
    if not available_datasets:
        print(" No datasets found! Please check the paths.")
        return None, None, None
    
    print(f"\n📊 Processing {len(available_datasets)} available datasets...")
    print("-" * 80)
    
    # Load each available dataset
    for dataset_name in available_datasets:
        path = DATASET_PATHS[dataset_name]
        features, labels = load_dataset(dataset_name, path)
        
        if features:
            all_features.extend(features)
            all_labels.extend(labels)
            
            # Track statistics per dataset
            unique_emotions, counts = np.unique(labels, return_counts=True)
            dataset_stats[dataset_name] = {
                'total_samples': len(features),
                'emotions': dict(zip(unique_emotions, counts.tolist()))
            }
        
        print("-" * 40)
    
    if not all_features:
        print("No data loaded from any dataset!")
        return None, None, None
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # Print comprehensive statistics
    print("\n" + "="*80)
    print("📈 COMPREHENSIVE DATASET STATISTICS")
    print("="*80)
    
    print(f"Dataset Overview:")
    print(f"   Total samples: {len(X):,}")
    print(f"   Available datasets: {len(available_datasets)}")
    print(f"   Feature shape: {X.shape}")
    print(f"   Data size: {X.nbytes / (1024*1024):.2f} MB")
    
    # Overall emotion distribution
    unique_emotions, counts = np.unique(y, return_counts=True)
    print(f"\nOverall Emotion Distribution:")
    total_samples = len(y)
    
    for emotion, count in zip(unique_emotions, counts):
        percentage = (count / total_samples) * 100
        print(f"   {emotion.capitalize():>12}: {count:>5} samples ({percentage:>5.1f}%)")
    
    # Per-dataset statistics
    print(f"\n📊 Per-Dataset Statistics:")
    print(f"   {'Dataset':<12} {'Samples':<8} {'Emotions':<50}")
    print(f"   {'-'*12} {'-'*8} {'-'*50}")
    
    for dataset_name in available_datasets:
        if dataset_name in dataset_stats:
            stats = dataset_stats[dataset_name]
            emotions_str = ', '.join([f"{k}:{v}" for k, v in stats['emotions'].items()])
            if len(emotions_str) > 47:
                emotions_str = emotions_str[:44] + "..."
            print(f"   {dataset_name:<12} {stats['total_samples']:<8} {emotions_str}")
    
    # Dataset balance analysis
    max_count = max(counts)
    min_count = min(counts)
    balance_ratio = min_count / max_count
    
    print(f"\n⚖️  Dataset Balance Analysis:")
    print(f"   Most common: {unique_emotions[np.argmax(counts)]} ({max_count} samples)")
    print(f"   Least common: {unique_emotions[np.argmin(counts)]} ({min_count} samples)")
    print(f"   Balance ratio: {balance_ratio:.3f} (1.0 = perfectly balanced)")
    
    if balance_ratio < 0.2:
        print("   Dataset is highly imbalanced - consider data augmentation")
    elif balance_ratio < 0.5:
        print("  Dataset has moderate imbalance")
    else:
        print("  Dataset is reasonably balanced")
    
    return X, y, unique_emotions

def build_model(input_shape, num_classes):
    """Build enhanced neural network model"""
    model = Sequential()

    # First dense block
    model.add(Dense(1024, input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    # Second dense block
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.4))

    # Third dense block
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))

    # Fourth dense block
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    # Fifth dense block
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=0.0005)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model

def plot_training_history(history):
    """Plot and save training history"""
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning curve
    plt.subplot(1, 3, 3)
    epochs = range(1, len(history.history['accuracy']) + 1)
    plt.plot(epochs, history.history['accuracy'], 'b-', label='Training', linewidth=2)
    plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    plt.title('Learning Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, "training_history.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Training plots saved to {MODEL_SAVE_PATH}training_history.png")

def save_training_info(history, test_acc, test_loss, emotion_classes, training_time, dataset_stats):
    """Save comprehensive training information to JSON"""
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'emotion_classes': emotion_classes.tolist(),
        'training_time_minutes': training_time,
        'epochs': len(history.history['accuracy']),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'best_val_epoch': int(np.argmax(history.history['val_accuracy']) + 1),
        'dataset_statistics': dataset_stats,
        'datasets_used': list(DATASET_PATHS.keys()),
        'model_architecture': {
            'layers': ['Dense(1024)', 'BatchNorm', 'LeakyReLU', 'Dropout(0.5)',
                      'Dense(512)', 'BatchNorm', 'LeakyReLU', 'Dropout(0.4)',
                      'Dense(256)', 'BatchNorm', 'LeakyReLU', 'Dropout(0.3)',
                      'Dense(128)', 'BatchNorm', 'LeakyReLU', 'Dropout(0.2)',
                      'Dense(64)', 'BatchNorm', 'LeakyReLU', 'Dropout(0.1)',
                      'Dense(output, softmax)'],
            'optimizer': 'Adam(lr=0.0005)',
            'loss': 'categorical_crossentropy'
        }
    }
    
    with open(os.path.join(MODEL_SAVE_PATH, TRAINING_HISTORY_NAME), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"💾 Training info saved to {MODEL_SAVE_PATH}{TRAINING_HISTORY_NAME}")

def main():
    print("ADVANCED MULTI-DATASET SPEECH EMOTION RECOGNITION")
    print("="*80)
    print("Supporting 6 Major Emotion Datasets:")
    print("   • RAVDESS  • TESS     • CREMA-D")
    print("   • EMO-DB   • IEMOCAP  • SAVEE")
    print("="*80)
    
    # Check dataset availability
    print(f"\n🔍 Dataset Path Verification:")
    available_count = 0
    for dataset_name, path in DATASET_PATHS.items():
        exists = os.path.exists(path)
        status = "Found" if exists else "❌ Not Found"
        print(f"   {dataset_name:<10}: {status}")
        if exists:
            available_count += 1
    
    if available_count == 0:
        print("\nNo dataset paths found!")
        print("Please update the DATASET_PATHS dictionary with correct paths.")
        return
    
    print(f"\nFound {available_count}/6 datasets available for training")
    
    start_time = datetime.now()
    
    # Load and prepare data
    print("\n" + "="*80)
    print("LOADING AND PREPARING MULTI-DATASET")
    print("="*80)
    
    X_combined, y_combined, emotion_classes = create_combined_dataset()
    
    if X_combined is None:
        print(" Cannot proceed without data. Please check dataset paths.")
        return
    
    # Prepare dataset statistics
    unique_emotions, counts = np.unique(y_combined, return_counts=True)
    dataset_stats = {
        'total_samples': len(X_combined),
        'emotion_distribution': dict(zip(unique_emotions, counts.tolist())),
        'feature_shape': list(X_combined.shape),
        'datasets_used': [name for name, path in DATASET_PATHS.items() if os.path.exists(path)]
    }
    
    print("\n" + "="*80)
    print(" PREPARING DATA FOR TRAINING")
    print("="*80)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = to_categorical(le.fit_transform(y_combined))
    X_flat = X_combined.reshape(X_combined.shape[0], -1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y_encoded, test_size=0.2, random_state=42, stratify=y_combined
    )
    
    print(f"📊 Data Split Complete:")
    print(f"   Training samples: {X_train.shape[0]:,} ({X_train.shape[0]/len(X_flat)*100:.1f}%)")
    print(f"   Test samples: {X_test.shape[0]:,} ({X_test.shape[0]/len(X_flat)*100:.1f}%)")
    print(f"   Input features: {X_train.shape[1]:,}")
    print(f"   Output classes: {y_train.shape[1]}")
    print(f"   Memory usage: {(X_train.nbytes + y_train.nbytes)/(1024*1024):.1f} MB")
    
    print("\n" + "="*80)
    print("BUILDING ENHANCED NEURAL NETWORK")
    print("="*80)
    
    model = build_model(X_train.shape[1], y_train.shape[1])
    
    print(" Model Architecture Summary:")
    model.summary()
    
    total_params = model.count_params()
    print(f"\n Model Statistics:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {total_params:,}")
    print(f"   Model Size: ~{total_params * 4 / (1024*1024):.1f} MB")
    
    print("\n" + "="*80)
    print(" TRAINING MULTI-DATASET MODEL")
    print("="*80)
    print("This may take 15-45 minutes depending on your hardware...")
    print("Training with 100 epochs, batch size 32, 20% validation split")
    
    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.2,
        verbose=1
    )
    
    print("\n" + "="*80)
    print(" EVALUATING MODEL PERFORMANCE")
    print("="*80)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds() / 60
    
    print(f"TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"FINAL PERFORMANCE METRICS:")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Training Time: {training_time:.2f} minutes")
    print(f"   Best Validation Accuracy: {max(history.history['val_accuracy']):.4f} ({max(history.history['val_accuracy'])*100:.2f}%)")
    print(f"   Final Training Accuracy: {history.history['accuracy'][-1]:.4f} ({history.history['accuracy'][-1]*100:.2f}%)")
    
    print("\n" + "="*80)
    print(" SAVING MODEL AND COMPONENTS")
    print("="*80)
    
    # Save model
    model_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save label encoder
    encoder_path = os.path.join(MODEL_SAVE_PATH, ENCODER_NAME)
    with open(encoder_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"Label encoder saved to: {encoder_path}")
    
    # Save training plots
    plot_training_history(history)
    
    # Save training information
    save_training_info(history, test_acc, test_loss, emotion_classes, training_time, dataset_stats)
    
    print(f"\nALL COMPONENTS SAVED SUCCESSFULLY!")
    print(f" Model directory: {os.path.abspath(MODEL_SAVE_PATH)}")
    print(f" Ready to run conversational AI: python run_ai.py")
    
    print(f"\nTRAINED EMOTION CLASSES:")
    print(f"{'='*50}")
    for i, emotion in enumerate(le.classes_):
        count = dataset_stats['emotion_distribution'].get(emotion, 0)
        print(f"   {i:2d}: {emotion.capitalize():<15} ({count:>5} samples)")
    
    print(f"\n📊 DATASETS SUCCESSFULLY INTEGRATED:")
    for dataset_name in dataset_stats['datasets_used']:
        print(f"    {dataset_name}")

if __name__ == "__main__":
    main()

