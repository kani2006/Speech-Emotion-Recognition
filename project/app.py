from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import librosa
import numpy as np
import pickle
import requests
import json
from tensorflow.keras.models import load_model
import tempfile
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = 'models/enhanced_emotion_model.h5'
ENCODER_PATH = 'models/label_encoder.pkl'
GEMINI_API_KEY = 'AIzaSyB5-aUDuKck8-EQHRTfVzfL2veaIBO3koA'  # Replace with your actual API key
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'

# Global variables for model and encoder
emotion_model = None
label_encoder = None

def load_models():
    """Load the trained emotion recognition model and label encoder"""
    global emotion_model, label_encoder
    
    try:
        if os.path.exists(MODEL_PATH):
            emotion_model = load_model(MODEL_PATH)
            logger.info("✅ Emotion recognition model loaded successfully")
        else:
            logger.error(f"❌ Model file not found: {MODEL_PATH}")
            return False
            
        if os.path.exists(ENCODER_PATH):
            with open(ENCODER_PATH, 'rb') as f:
                label_encoder = pickle.load(f)
            logger.info("✅ Label encoder loaded successfully")
        else:
            logger.error(f"❌ Encoder file not found: {ENCODER_PATH}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        return False

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
        return mfccs.flatten()
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return None

def predict_emotion_from_audio(audio_file):
    """Predict emotion from audio file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_file.save(tmp_file.name)
            
            # Extract features
            features = extract_features(tmp_file.name)
            if features is None:
                return None, 0.0
            
            # Reshape for prediction
            features = features.reshape(1, -1)
            
            # Predict
            prediction = emotion_model.predict(features)
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction))
            
            # Get emotion label
            emotion = label_encoder.inverse_transform([predicted_class])[0]
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            return emotion, confidence
            
    except Exception as e:
        logger.error(f"Error predicting emotion from audio: {str(e)}")
        return None, 0.0

def predict_emotion_from_text(text):
    """Predict emotion from text using simple keyword matching"""
    emotion_keywords = {
        'happy': ['happy', 'joy', 'excited', 'cheerful', 'delighted', 'pleased', 'glad', 'wonderful', 'amazing', 'great', 'fantastic', 'awesome', 'love', 'excellent'],
        'sad': ['sad', 'depressed', 'unhappy', 'miserable', 'sorrowful', 'gloomy', 'down', 'blue', 'melancholy', 'crying', 'tears', 'heartbroken'],
        'angry': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'rage', 'hate', 'frustrated', 'outraged', 'pissed', 'livid', 'enraged'],
        'fearful': ['scared', 'afraid', 'frightened', 'terrified', 'anxious', 'worried', 'nervous', 'panic', 'fear', 'stressed', 'overwhelmed'],
        'surprised': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'bewildered', 'wow', 'incredible', 'unbelievable'],
        'disgust': ['disgusted', 'revolted', 'repulsed', 'sick', 'nauseated', 'gross', 'yuck', 'awful', 'terrible', 'horrible'],
        'neutral': ['okay', 'fine', 'normal', 'regular', 'usual', 'standard', 'alright', 'whatever']
    }
    
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, keywords in emotion_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score
    
    if emotion_scores:
        predicted_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = min(emotion_scores[predicted_emotion] / len(text.split()), 1.0)
        return predicted_emotion, max(confidence, 0.3)  # Minimum confidence of 30%
    
    return 'neutral', 0.5

def generate_gemini_response(emotion, confidence, text=None, is_audio=False):
    """Generate empathetic response using Gemini 2.0 Flash API based on detected emotion"""
    
    # Create context-aware prompt
    if is_audio:
        prompt = f"""You are an empathetic AI assistant specialized in emotional support. A user has just shared an audio message, and I've detected their emotional state as '{emotion}' with {confidence*100:.1f}% confidence.

Please provide a compassionate and appropriate response that:
1. Acknowledges their emotional state with empathy
2. Shows genuine understanding and validation
3. Offers appropriate support, encouragement, or comfort based on the emotion
4. Asks a thoughtful follow-up question to encourage further conversation
5. Keep the response warm, conversational, and between 2-3 sentences
6. Use appropriate emojis to convey empathy

Detected emotion: {emotion}
Confidence level: {confidence*100:.1f}%
Input type: Audio message

Generate an empathetic and supportive response:"""
    else:
        prompt = f"""You are an empathetic AI assistant specialized in emotional support. A user has written the following message: "{text}"

I've analyzed their emotional state as '{emotion}' with {confidence*100:.1f}% confidence.

Please provide a compassionate and appropriate response that:
1. Acknowledges both their message content and emotional state
2. Shows genuine empathy and understanding
3. Validates their feelings appropriately
4. Offers relevant support, encouragement, or comfort based on the detected emotion
5. Asks a thoughtful follow-up question to continue the meaningful conversation
6. Keep the response warm, conversational, and between 2-3 sentences
7. Use appropriate emojis to convey empathy and warmth

User's message: "{text}"
Detected emotion: {emotion}
Confidence level: {confidence*100:.1f}%

Generate an empathetic and supportive response:"""
    
    try:
        # Prepare the request payload for Gemini 2.0 Flash
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 200,
                "stopSequences": []
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        # Set up headers
        headers = {
            'Content-Type': 'application/json',
        }
        
        # Make the API request
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Extract the generated text from the response
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                candidate = response_data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    generated_text = candidate['content']['parts'][0]['text']
                    return generated_text.strip()
            
            # If we can't extract the text, fall back to default
            logger.warning("Could not extract text from Gemini response")
            
        else:
            logger.error(f"Gemini API error: {response.status_code} - {response.text}")
            
    except requests.exceptions.Timeout:
        logger.error("Gemini API request timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"Gemini API request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating Gemini response: {str(e)}")
    
    # Fallback responses based on emotion if API fails
    fallback_responses = {
        'happy': "I can feel the joy in your message! 😊 It's wonderful to see you in such a positive mood. What's been bringing you so much happiness lately?",
        'sad': "I can sense you're going through a difficult time, and I want you to know that your feelings are completely valid. 💙 I'm here to listen - would you like to share what's been weighing on your heart?",
        'angry': "I can hear the frustration in your words, and it's completely understandable to feel this way. 😤 Take a deep breath with me - what's been causing you to feel so upset?",
        'fearful': "I can sense some anxiety and worry in what you've shared, and that must feel really overwhelming. 🤗 You're not alone in this - what's been making you feel so uneasy?",
        'surprised': "Wow, it sounds like something really unexpected happened! ✨ I can feel your amazement - I'd love to hear more about what caught you off guard!",
        'disgust': "It seems like something has really bothered or repulsed you, and those feelings are completely valid. 😔 Sometimes we encounter things that just don't sit right with us - want to talk about what happened?",
        'neutral': "Thanks for sharing with me! I'm here and ready to listen to whatever's on your mind. 🌟 How has your day been treating you so far?",
        'calm': "You seem very peaceful and centered right now, which is beautiful to witness. 🌸 I'd love to know what's been helping you feel so balanced and serene today?"
    }
    
    return fallback_responses.get(emotion, "I'm here to listen and support you through whatever you're feeling right now. 💝 How can I help you today?")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': emotion_model is not None,
        'encoder_loaded': label_encoder is not None,
        'gemini_model': 'gemini-2.0-flash',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict/audio', methods=['POST'])
def predict_audio_emotion():
    """Predict emotion from uploaded audio file"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Predict emotion
        emotion, confidence = predict_emotion_from_audio(audio_file)
        
        if emotion is None:
            return jsonify({'error': 'Failed to process audio file'}), 500
        
        # Generate AI response using Gemini 2.0 Flash
        ai_response = generate_gemini_response(emotion, confidence, is_audio=True)
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': confidence,
            'ai_response': ai_response,
            'timestamp': datetime.now().isoformat(),
            'type': 'audio',
            'model_used': 'gemini-2.0-flash'
        })
        
    except Exception as e:
        logger.error(f"Error in audio prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict/text', methods=['POST'])
def predict_text_emotion():
    """Predict emotion from text input"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Predict emotion from text
        emotion, confidence = predict_emotion_from_text(text)
        
        # Generate AI response using Gemini 2.0 Flash
        ai_response = generate_gemini_response(emotion, confidence, text=text, is_audio=False)
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': confidence,
            'ai_response': ai_response,
            'original_text': text,
            'timestamp': datetime.now().isoformat(),
            'type': 'text',
            'model_used': 'gemini-2.0-flash'
        })
        
    except Exception as e:
        logger.error(f"Error in text prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """General chat endpoint that handles both text and audio"""
    try:
        # Check if it's an audio file
        if 'audio' in request.files:
            return predict_audio_emotion()
        
        # Otherwise, treat as text
        return predict_text_emotion()
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/test-gemini', methods=['GET'])
def test_gemini():
    """Test endpoint to verify Gemini 2.0 Flash API connectivity"""
    try:
        test_response = generate_gemini_response('happy', 0.8, text="I'm feeling great today!", is_audio=False)
        return jsonify({
            'success': True,
            'test_response': test_response,
            'model': 'gemini-2.0-flash',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Gemini test failed: {str(e)}")
        return jsonify({'error': f'Gemini API test failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        logger.info("🚀 Starting Flask application with Gemini 2.0 Flash...")
        logger.info(f"🤖 Using Gemini API URL: {GEMINI_API_URL}")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("❌ Failed to load models. Please check model files.")

