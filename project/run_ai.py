import os
import re
import librosa
import numpy as np
import sounddevice as sd
import wavio
import soundfile as sf
import speech_recognition as sr
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import random
from datetime import datetime
import torch
import json
import pickle
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
MODEL_SAVE_PATH = 'models/'
MODEL_NAME = 'enhanced_emotion_model.h5'
ENCODER_NAME = 'label_encoder.pkl'
def check_model_files():
    model_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
    encoder_path = os.path.join(MODEL_SAVE_PATH, ENCODER_NAME)
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please run 'python train_model.py' first to train the model.")
        return False
    
    if not os.path.exists(encoder_path):
        print(f"Label encoder not found: {encoder_path}")
        print("Please run 'python train_model.py' first to train the model.")
        return False
    
    return True
def load_trained_model():
    try:
        model_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
        encoder_path = os.path.join(MODEL_SAVE_PATH, ENCODER_NAME)
        model = load_model(model_path)
        
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, label_encoder
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
        
def extract_features(file_path, max_pad_len=174):
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
class EmotionalConversationalAI:
    def __init__(self):
        print("🤖 Initializing Enhanced Emotional Conversational AI...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            self.llm_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
            self.tokenizer.pad_token = self.tokenizer.eos_token

            print("DialoGPT model loaded successfully!")
        except Exception as e:
            print(f"Error loading DialoGPT: {e}")
            print("Falling back to Flan-T5...")
            try:
                self.llm_model = pipeline("text2text-generation", model="google/flan-t5-base")
                self.tokenizer = None
                print("Flan-T5 model loaded successfully!")
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                self.llm_model = None
                self.tokenizer = None
        
        # Conversation memory
        self.conversation_history = []
        self.emotion_history = []
        self.user_profile = {
            'dominant_emotions': {},
            'topics_discussed': [],
            'conversation_style': 'balanced',
            'session_start': datetime.now(),
            'total_interactions': 0,
            'emotion_transitions': [],
            'preferred_topics': {}
        }
        
        self.emotion_prompts = {
            'happy': {
                'system_prompt': "You are an enthusiastic, joyful AI companion who celebrates positive moments and encourages continued happiness. Use upbeat language and share in the user's joy.",
                'conversation_starters': [
                    "That's wonderful! Tell me more about what's making you so happy!",
                    "Your positive energy is contagious! What else is going well in your life?",
                    "I love hearing about good news! How are you planning to celebrate?",
                    "Your happiness is beautiful! What's the best part about this moment?"
                ],
                'follow_ups': [
                    "How does this happiness compare to other great moments you've had?",
                    "What do you think contributed most to this positive feeling?",
                    "Who would you most like to share this joy with?",
                    "What would make this moment even more special?"
                ]
            },
            'sad': {
                'system_prompt': "You are a compassionate, gentle AI companion who provides comfort and support during difficult times. Use warm, understanding language and validate emotions.",
                'conversation_starters': [
                    "I can hear that you're going through a tough time. Would you like to talk about what's bothering you?",
                    "It's okay to feel sad. Sometimes talking helps. I'm here to listen.",
                    "Your feelings are completely valid. What's been weighing on your heart?",
                    "I'm here with you in this difficult moment. What's troubling you?"
                ],
                'follow_ups': [
                    "How long have you been feeling this way?",
                    "Have you been able to talk to anyone else about this?",
                    "What usually helps you feel a little better when you're down?",
                    "Would you like to share what's making this particularly hard right now?"
                ]
            },
            'angry': {
                'system_prompt': "You are a calm, understanding AI companion who helps process anger constructively. Acknowledge the emotion while guiding toward solutions.",
                'conversation_starters': [
                    "I can sense your frustration. What's really getting under your skin?",
                    "That sounds incredibly frustrating! Tell me what happened.",
                    "Your anger is completely understandable. What triggered these feelings?",
                    "I hear the intensity in your voice. What's making you feel this way?"
                ],
                'follow_ups': [
                    "What would need to change for you to feel better about this situation?",
                    "Have you encountered similar frustrations before?",
                    "What would be the most satisfying way to resolve this?",
                    "How would you ideally like to see this situation handled?"
                ]
            },
            'fearful': {
                'system_prompt': "You are a reassuring, supportive AI companion who helps build confidence and addresses fears with understanding and encouragement.",
                'conversation_starters': [
                    "I can sense some worry in your voice. What's been on your mind?",
                    "It takes courage to share when you're feeling anxious. What's concerning you?",
                    "Fear is a natural emotion. What's making you feel uncertain right now?",
                    "I'm here to support you through this worry. What's troubling you?"
                ],
                'follow_ups': [
                    "What's the worst-case scenario you're imagining?",
                    "Have you faced similar fears before? How did you handle them?",
                    "What would help you feel more secure about this situation?",
                    "What support do you need to feel more confident about this?"
                ]
            },
            'surprised': {
                'system_prompt': "You are a curious, engaged AI companion who shares in wonder and helps process unexpected experiences.",
                'conversation_starters': [
                    "Wow! I can hear the amazement in your voice! What surprised you?",
                    "That sounds like quite a revelation! Tell me all about it!",
                    "I love when life throws us pleasant surprises! What happened?",
                    "Your surprise is infectious! What caught you off guard?"
                ],
                'follow_ups': [
                    "How are you processing this unexpected news?",
                    "What was your first reaction when this happened?",
                    "How might this change things for you going forward?",
                    "What's the most surprising part about this whole situation?"
                ]
            },
            'disgust': {
                'system_prompt': "You are an understanding AI companion who validates strong reactions while helping find constructive ways forward.",
                'conversation_starters': [
                    "I can tell something really bothered you. What happened?",
                    "That sounds like it was really unpleasant. Want to talk about it?",
                    "Sometimes we encounter things that just don't sit right with us. What's going on?",
                    "I hear that something really didn't agree with you. What was it?"
                ],
                'follow_ups': [
                    "What specifically bothered you most about this situation?",
                    "How do you usually handle situations that make you uncomfortable?",
                    "What would need to change for you to feel better about this?",
                    "How can you protect yourself from similar situations in the future?"
                ]
            },
            'neutral': {
                'system_prompt': "You are a balanced, thoughtful AI companion who engages in meaningful conversation and helps explore thoughts and feelings.",
                'conversation_starters': [
                    "I'm here and ready to listen. What's on your mind today?",
                    "How are you feeling right now? I'm interested in your perspective.",
                    "What would you like to talk about? I'm here for whatever you need.",
                    "I'm curious about your thoughts today. What's been occupying your mind?"
                ],
                'follow_ups': [
                    "How has your day been treating you?",
                    "What's been occupying your thoughts lately?",
                    "Is there anything you've been wanting to discuss?",
                    "What's been the most interesting part of your day so far?"
                ]
            },
            'calm': {
                'system_prompt': "You are a peaceful, mindful AI companion who appreciates tranquil moments and engages in thoughtful, serene conversation.",
                'conversation_starters': [
                    "There's something beautifully peaceful about your energy. How are you feeling?",
                    "I appreciate the calm presence you bring. What's contributing to this serenity?",
                    "Your tranquil energy is lovely. What's been bringing you peace lately?",
                    "I sense a wonderful calmness in you. What's creating this peaceful state?"
                ],
                'follow_ups': [
                    "What helps you maintain this sense of calm?",
                    "How do you find peace in your daily life?",
                    "What does inner peace mean to you?",
                    "What practices help you stay centered like this?"
                ]
            }
        }
        
        self.emotion_emojis = {
            'happy': ['😊', '😄', '🎉', '✨', '🌟', '💫', '🎈', '🌈', '🥳', '😁'],
            'sad': ['💙', '🤗', '🌙', '💜', '🕊️', '🌸', '☁️', '💝', '🫂', '💙'],
            'angry': ['🔥', '💪', '⚡', '🌋', '🎯', '🛡️', '⭐', '🌊', '💢', '🔴'],
            'fearful': ['🌟', '🛡️', '💎', '🌅', '🦋', '🌱', '💫', '🏔️', '🌈', '💪'],
            'surprised': ['🎊', '✨', '🎆', '💥', '🌟', '🎭', '🎪', '🎨', '😲', '🤯'],
            'disgust': ['🌿', '🧹', '✨', '🌸', '💎', '🌊', '🕊️', '🌈', '🧼', '🌱'],
            'neutral': ['🌸', '🍃', '🌙', '💫', '🌊', '🎋', '🌅', '🕊️', '🌿', '☁️'],
            'calm': ['🌸', '🍃', '🌙', '💙', '🌊', '☁️', '🕊️', '🌿', '🧘', '🌅']
        }
    def update_user_profile(self, emotion, topics):
        """Enhanced user profile tracking"""
        self.user_profile['total_interactions'] += 1
        
        if emotion in self.user_profile['dominant_emotions']:
            self.user_profile['dominant_emotions'][emotion] += 1
        else:
            self.user_profile['dominant_emotions'][emotion] = 1
        for topic in topics:
            if topic in self.user_profile['preferred_topics']:
                self.user_profile['preferred_topics'][topic] += 1
            else:
                self.user_profile['preferred_topics'][topic] = 1
        
        # Track emotion transitions
        if len(self.emotion_history) > 0:
            prev_emotion = self.emotion_history[-1]['primary_emotion']
            if prev_emotion != emotion:
                transition = f"{prev_emotion}→{emotion}"
                self.user_profile['emotion_transitions'].append(transition)
        
        emotion_counts = self.user_profile['dominant_emotions']
        if emotion_counts:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            if dominant_emotion in ['happy', 'surprised']:
                self.user_profile['conversation_style'] = 'enthusiastic'
            elif dominant_emotion in ['sad', 'fearful']:
                self.user_profile['conversation_style'] = 'supportive'
            elif dominant_emotion in ['angry', 'disgust']:
                self.user_profile['conversation_style'] = 'solution_focused'
            else:
                self.user_profile['conversation_style'] = 'balanced'
    def generate_contextual_prompt(self, current_emotion, user_text, conversation_turn):
        emotion_data = self.emotion_prompts.get(current_emotion, self.emotion_prompts['neutral'])
        
        context = ""
        if self.conversation_history:
            recent_context = self.conversation_history[-3:]
            context = "Previous conversation:\n"
            for exchange in recent_context:
                context += f"User: {exchange['user']}\nAI: {exchange['ai']}\n"
            context += "\n"
        
        # Analyze emotion patterns
        emotion_pattern = ""
        if len(self.emotion_history) > 1:
            recent_emotions = [e['primary_emotion'] for e in self.emotion_history[-3:]]
            emotion_pattern = f"Recent emotion pattern: {' → '.join(recent_emotions)}\n"
        
        # Choose appropriate response type
        if conversation_turn <= 1:
            prompt_template = random.choice(emotion_data['conversation_starters'])
        else:
            prompt_template = random.choice(emotion_data['follow_ups'])
        
        # Build enhanced prompt
        full_prompt = f"""
{emotion_data['system_prompt']}
{context}{emotion_pattern}
Current user emotion: {current_emotion}
User just said: "{user_text}"
User profile:
- Conversation style: {self.user_profile['conversation_style']}
- Total interactions: {self.user_profile['total_interactions']}
- Session duration: {(datetime.now() - self.user_profile['session_start']).seconds // 60} minutes
- Dominant emotions: {list(self.user_profile['dominant_emotions'].keys())[:3]}
Respond in a way that:
1. Acknowledges their {current_emotion} emotion authentically
2. Continues the conversation naturally and engagingly
3. Shows genuine interest and empathy
4. Asks thoughtful follow-up questions
5. Matches their emotional energy appropriately
6. Builds on previous conversation context
Response:"""
        
        return full_prompt
    def generate_llm_response(self, emotion, user_text, conversation_turn):
        """Generate response using LLM with enhanced emotional context"""
        if self.llm_model is None:
            return self.get_fallback_response(emotion, user_text)
        
        try:
            prompt = self.generate_contextual_prompt(emotion, user_text, conversation_turn)
            
            if self.tokenizer:  # Using DialoGPT or similar model
                # Encode past conversation history if exists
                chat_history_ids = None
                if self.conversation_history:
                    chat_text = ""
                    for exchange in self.conversation_history[-5:]:
                        chat_text += exchange['user'] + self.tokenizer.eos_token
                        chat_text += exchange['ai'] + self.tokenizer.eos_token
                    chat_history_ids = self.tokenizer.encode(chat_text, return_tensors='pt')
                
                # Tokenize current user input with attention mask
                inputs = self.tokenizer(
                    user_text + self.tokenizer.eos_token,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )
                new_user_input_ids = inputs['input_ids']
                attention_mask_new = inputs['attention_mask']

                # Concatenate conversation history with new user input
                if chat_history_ids is not None:
                    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
                    attention_mask_history = torch.ones(chat_history_ids.shape, dtype=torch.long)
                    combined_attention_mask = torch.cat([attention_mask_history, attention_mask_new], dim=-1)
                else:
                    bot_input_ids = new_user_input_ids
                    combined_attention_mask = attention_mask_new
    
               # Inside generate_llm_response, before calling generate()

                max_length = min(bot_input_ids.shape[-1] + 150, self.llm_model.config.max_position_embeddings)  # Compute max length safely

                with torch.no_grad():
                    chat_history_ids = self.llm_model.generate(
                        bot_input_ids,
                        attention_mask=combined_attention_mask,
                        max_length=max_length,   # Use computed max_length here
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.8,
                        repetition_penalty=1.2,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

            
                response = self.tokenizer.decode(
                    chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                    skip_special_tokens=True
                )
                
            else:
                # For fallback model like Flan-T5
                result = self.llm_model(prompt, max_length=150, do_sample=True, temperature=0.8, top_p=0.9)
                response = result[0]['generated_text']
            
            response = self.clean_response(response, emotion)
            return response
        
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return self.get_fallback_response(emotion, user_text)        
            
    def clean_response(self, response, emotion):
        """Clean and enhance the LLM response"""
        response = response.strip()
        
        # Remove unwanted patterns
        response = re.sub(r'^(AI:|Assistant:|Response:)', '', response).strip()
        
        # Add appropriate emoji
        emoji = random.choice(self.emotion_emojis.get(emotion, ['💫']))
        
        # Ensure response isn't too long
        if len(response) > 200:
            sentences = response.split('.')
            response = '. '.join(sentences[:2]) + '.'
        
        # Add emoji if not present
        emoji_chars = ''.join([char for emoji_list in self.emotion_emojis.values() for char in emoji_list])
        if not any(char in response for char in emoji_chars):
            response = f"{response} {emoji}"
        
        return response
    def get_fallback_response(self, emotion, user_text):
        """Enhanced fallback response if LLM fails"""
        emotion_data = self.emotion_prompts.get(emotion, self.emotion_prompts['neutral'])
        response = random.choice(emotion_data['conversation_starters'])
        emoji = random.choice(self.emotion_emojis.get(emotion, ['💫']))
        return f"{response} {emoji}"
    def continue_conversation(self, audio_emotion, text_emotion, transcribed_text, keywords):
        """Main method to continue conversation based on emotions"""
        # Determine primary emotion with confidence weighting
        primary_emotion = text_emotion if text_emotion != 'unknown' else audio_emotion
        if primary_emotion == 'unknown':
            primary_emotion = 'neutral'
        
        # Update user profile
        self.update_user_profile(primary_emotion, keywords)
        
        # Track emotion history
        self.emotion_history.append({
            'timestamp': datetime.now(),
            'audio_emotion': audio_emotion,
            'text_emotion': text_emotion,
            'primary_emotion': primary_emotion,
            'keywords': keywords
        })
        
        # Generate LLM response
        conversation_turn = len(self.conversation_history) + 1
        llm_response = self.generate_llm_response(primary_emotion, transcribed_text, conversation_turn)
        
        # Store conversation
        self.conversation_history.append({
            'user': transcribed_text,
            'ai': llm_response,
            'emotion': primary_emotion,
            'keywords': keywords,
            'timestamp': datetime.now().isoformat()
        })
        
        return llm_response, primary_emotion
    def get_conversation_insights(self):
        """Provide comprehensive insights about the conversation"""
        if not self.emotion_history:
            return "No conversation data yet."
        
        # Emotion distribution
        emotions = [e['primary_emotion'] for e in self.emotion_history]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
        
        # Conversation metrics
        duration = (datetime.now() - self.user_profile['session_start']).seconds // 60
        
        # Topic analysis
        top_topics = sorted(self.user_profile['preferred_topics'].items(), 
                           key=lambda x: x[1], reverse=True)[:5]
        
        # Emotion transitions
        unique_transitions = list(set(self.user_profile['emotion_transitions']))
        
        insights = f"""
📊 COMPREHENSIVE CONVERSATION INSIGHTS
{'='*50}
⏱️  SESSION METRICS:
• Duration: {duration} minutes
• Total exchanges: {len(self.conversation_history)}
• Total interactions: {self.user_profile['total_interactions']}
🎭 EMOTIONAL ANALYSIS:
• Dominant emotion: {dominant_emotion.upper()}
• Conversation style: {self.user_profile['conversation_style']}
• Emotions detected: {', '.join(emotion_counts.keys())}
• Emotion distribution: {dict(emotion_counts)}
🔄 EMOTION TRANSITIONS:
• Unique patterns: {len(unique_transitions)}
• Recent transitions: {unique_transitions[-3:] if unique_transitions else 'None'}
💬 TOPIC ANALYSIS:
• Top discussed topics: {[topic for topic, count in top_topics]}
• Topic engagement: {dict(top_topics)}
🧠 AI INSIGHTS:
• User tends to be: {dominant_emotion}
• Conversation flows: {'Dynamic' if len(unique_transitions) > 3 else 'Stable'}
• Engagement level: {'High' if len(self.conversation_history) > 5 else 'Moderate'}
"""
        return insights
# -----------------------------
# 4️⃣ AUDIO PROCESSING FUNCTIONS
# -----------------------------
def record_voice(filename="my_voice.wav", duration=5, fs=22050):
    """Record voice with enhanced feedback"""
    print(f"🎤 Recording for {duration} seconds... Speak now!")
    print("💡 Tip: Speak clearly and express your emotions naturally")
    
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    
    # Show countdown
    for i in range(duration, 0, -1):
        print(f"⏰ {i}...", end=" ", flush=True)
        sd.wait(1000)  # Wait 1 second
    
    sd.wait()
    wavio.write(filename, audio, fs, sampwidth=2)
    print(f"\n✅ Recording saved as {filename}")
    return filename
def convert_to_wav(filepath):
    """Convert audio file to WAV format"""
    if filepath.lower().endswith(".mp3"):
        print("🔄 Converting MP3 to WAV...")
        audio, sr = librosa.load(filepath, sr=None)
        wav_path = filepath.rsplit(".", 1)[0] + ".wav"
        sf.write(wav_path, audio, sr)
        print("✅ Conversion complete!")
        return wav_path
    return filepath
def choose_input():
    """Enhanced input selection with better UI"""
    print("\n" + "="*60)
    print("🤖 ENHANCED EMOTIONAL CONVERSATIONAL AI WITH LLM 🤖")
    print("="*60)
    print("📊 Using trained RAVDESS + TESS model for emotion recognition!")
    print("="*60)
    
    choice = input("Choose input method:\n1️⃣  Record your voice\n2️⃣  Upload audio file\n\nYour choice: ").strip()
    
    if choice == "1":
        duration = input("Recording duration in seconds (default 5): ").strip()
        duration = int(duration) if duration.isdigit() else 5
        filepath = record_voice(duration=duration)
    elif choice == "2":
        filepath = input("Enter audio file path (.wav/.mp3): ").strip()
        if os.path.exists(filepath):
            filepath = convert_to_wav(filepath)
        else:
            print("❌ File not found!")
            return choose_input()
    else:
        print("❌ Invalid choice!")
        return choose_input()
    return filepath
def predict_emotion(filepath, model, label_encoder, max_pad_len=174):
    try:
        mfccs = extract_features(filepath, max_pad_len)
        if mfccs is None:
            return "unknown", 0.0
        
        mfccs_flat = mfccs.reshape(1, -1)
        preds = model.predict(mfccs_flat, verbose=0)
        idx = np.argmax(preds)
        confidence = preds[0][idx]
        emotion = label_encoder.inverse_transform([idx])[0]
        
        return emotion, confidence
    except Exception as e:
        print(f"Error in emotion prediction: {e}")
        return "unknown", 0.0
recognizer = sr.Recognizer()
def speech_to_text(filepath):
    try:
        with sr.AudioFile(filepath) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
        
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}")
        return ""
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return ""
try:
    nlp_emotion = pipeline("text-classification",
                          model="j-hartmann/emotion-english-distilroberta-base",
                          top_k=None)
    print("Text emotion analyzer loaded successfully!")
except Exception as e:
    print(f"Error loading text emotion analyzer: {e}")
    nlp_emotion = None
def analyze_text_emotion(text):
    """Enhanced text emotion analysis"""
    if not text.strip():
        return "unknown", 0.0
    
    if nlp_emotion is None:
        return "unknown", 0.0
    
    try:
        results = nlp_emotion(text)[0]
        top = max(results, key=lambda x: x['score'])
        
        emotion_mapping = {
            'joy': 'happy',
            'fear': 'fearful',
            'surprise': 'surprised'
        }
        
        emotion = emotion_mapping.get(top['label'], top['label'])
        return emotion, top['score']
    except Exception as e:
        print(f"Text emotion analysis error: {e}")
        return "unknown", 0.0
def extract_keywords(text, top_n=5):
    """Enhanced keyword extraction"""
    if not text:
        return []
    
    stopwords = set([
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "of", "and",
        "to", "for", "i", "you", "it", "this", "that", "with", "as", "but", "my", "me",
        "we", "they", "them", "their", "our", "your", "his", "her", "him", "she", "he",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "can", "shall", "am", "so", "if", "or",
        "not", "no", "yes", "well", "just", "now", "then", "here", "there", "when", "where",
        "why", "how", "what", "who", "which", "than", "too", "very", "much", "many", "more",
        "most", "some", "any", "all", "both", "each", "few", "other", "another", "such", "only"
    ])
    
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    freq = {}
    for w in keywords:
        freq[w] = freq.get(w, 0) + 1
    
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]
def display_emotion_analysis(audio_emotion, audio_conf, text_emotion, text_conf, transcribed_text, keywords):
    print("\n" + "="*70)
    print("COMPREHENSIVE EMOTION ANALYSIS RESULTS")
    print("="*70)
    
    print(f"Audio Emotion: {audio_emotion.upper()} ({audio_conf*100:.1f}% confidence)")
    print(f"Text Emotion: {text_emotion.upper()} ({text_conf*100:.1f}% confidence)")
    print(f"Transcribed Text: \"{transcribed_text}\"")
    print(f"Keywords: {', '.join(keywords) if keywords else 'None detected'}")
    
    # Determine dominant emotion with logic
    if text_conf > audio_conf and text_emotion != 'unknown':
        dominant = text_emotion
        source = "Text Content"
        confidence = text_conf
    elif audio_emotion != 'unknown':
        dominant = audio_emotion
        source = "Voice Tone"
        confidence = audio_conf
    else:
        dominant = 'neutral'
        source = "Default"
        confidence = 0.5
    
    print(f"Dominant Emotion: {dominant.upper()} (Source: {source}, {confidence*100:.1f}%)")
    
    # Emotion interpretation
    emotion_descriptions = {
        'happy': '😊 Positive, joyful, content',
        'sad': '😢 Melancholic, down, sorrowful',
        'angry': '😠 Frustrated, irritated, upset',
        'fearful': '😰 Anxious, worried, concerned',
        'surprised': '😲 Amazed, shocked, unexpected',
        'disgust': '🤢 Repulsed, uncomfortable, averse',
        'neutral': '😐 Balanced, calm, steady',
        'calm': '😌 Peaceful, relaxed, serene'
    }
    
    description = emotion_descriptions.get(dominant, 'Complex emotional state')
    print(f"💭 Interpretation: {description}")
    print("="*70)
def display_conversation_header(turn_number):
    print(f"\n{'='*70}")
    print(f"CONVERSATION TURN {turn_number}")
    print(f"{'='*70}")
def display_model_info():
    try:
        history_path = os.path.join(MODEL_SAVE_PATH, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                training_info = json.load(f)
            
            print(f"\nMODEL INFORMATION:")
            print(f"{'='*50}")
            print(f"Training Date: {training_info.get('timestamp', 'Unknown')[:10]}")
            print(f"Test Accuracy: {training_info.get('test_accuracy', 0)*100:.2f}%")
            print(f"Emotion Classes: {', '.join(training_info.get('emotion_classes', []))}")
            print(f"Training Epochs: {training_info.get('epochs', 'Unknown')}")
            print(f"Best Validation Accuracy: {training_info.get('best_val_accuracy', 0)*100:.2f}%")
            print(f"{'='*50}")
    except Exception as e:
        print(f"Could not load model info: {e}")
        
def main():
    print("Starting Enhanced Emotional Conversational AI...")
    print("="*60)
    
    if not check_model_files():
        return
    model, label_encoder = load_trained_model()
    if model is None or label_encoder is None:
        return
    
    display_model_info()
    
    conversational_ai = EmotionalConversationalAI()
    
    print("\nEnhanced Emotional Conversational AI with LLM is ready!")
    print("Using trained RAVDESS + TESS model for superior emotion recognition!")
    print("Features advanced conversation memory and emotional intelligence!")
    print("\nCommands:")
    print("   • 'insights' - View detailed conversation analysis")
    print("   • 'profile' - View your emotional profile")
    print("   • 'history' - View conversation history")
    print("   • 'quit' - Exit the application")
    
    conversation_count = 0
    
    while True:
        try:
            conversation_count += 1
            display_conversation_header(conversation_count)
            
            user_input = input("\nChoose: [1] Record voice [2] Upload file [3] 'insights' [4] 'profile' [5] 'history' [6] 'quit': ").strip()
            
            if user_input.lower() == 'quit':
                print("\nThank you for the wonderful conversation!")
                print(conversational_ai.get_conversation_insights())
                break
            elif user_input.lower() == 'insights':
                print(conversational_ai.get_conversation_insights())
                continue
            elif user_input.lower() == 'profile':
                print(f"\nYOUR EMOTIONAL PROFILE:")
                print(f"{'='*50}")
                print(f"Dominant emotions: {conversational_ai.user_profile['dominant_emotions']}")
                print(f"Conversation style: {conversational_ai.user_profile['conversation_style']}")
                print(f"Total interactions: {conversational_ai.user_profile['total_interactions']}")
                print(f"Preferred topics: {list(conversational_ai.user_profile['preferred_topics'].keys())[:5]}")
                continue
            elif user_input.lower() == 'history':
                print(f"\nCONVERSATION HISTORY:")
                print(f"{'='*50}")
                for i, exchange in enumerate(conversational_ai.conversation_history[-5:], 1):
                    print(f"{i}. [{exchange['emotion'].upper()}] You: {exchange['user']}")
                    print(f"   AI: {exchange['ai']}\n")
                continue
            elif user_input in ['1', '2']:
                if user_input == '1':
                    duration = input("Recording duration (default 5s): ").strip()
                    duration = int(duration) if duration.isdigit() else 5
                    filename = record_voice(duration=duration)
                else:
                    filename = input("Enter audio file path: ").strip()
                    if not os.path.exists(filename):
                        print("File not found!")
                        continue
                    filename = convert_to_wav(filename)
            else:
                print("Invalid choice!")
                continue
            
            print("\nAnalyzing emotions...")
            audio_emotion, audio_conf = predict_emotion(filename, model, label_encoder)

            print("Converting speech to text...")
            transcribed_text = speech_to_text(filename)
            
            if not transcribed_text:
                print("Could not transcribe audio. Please try again with clearer speech.")
                continue
            
            print("Analyzing text emotion...")
            text_emotion, text_conf = analyze_text_emotion(transcribed_text)
            
            keywords = extract_keywords(transcribed_text)
            
            display_emotion_analysis(audio_emotion, audio_conf, text_emotion, text_conf, transcribed_text, keywords)
            
            print("\nAI is thinking and crafting a response...")
            llm_response, primary_emotion = conversational_ai.continue_conversation(
                audio_emotion, text_emotion, transcribed_text, keywords
            )
            
            print(f"\nAI RESPONSE (Emotion-aware: {primary_emotion.upper()}):")
            print("="*70)
            print(f"{llm_response}")
            print("="*70)
            
            if conversation_count > 1:
                recent_emotions = [e['primary_emotion'] for e in conversational_ai.emotion_history[-3:]]
                print(f"📈 Emotion Flow: {' → '.join(recent_emotions)}")
                print(f"Your Style: {conversational_ai.user_profile['conversation_style']}")
        
            duration = (datetime.now() - conversational_ai.user_profile['session_start']).seconds // 60
            print(f"Session: {duration}min | Exchanges: {len(conversational_ai.conversation_history)}")
            
        except KeyboardInterrupt:
            print("\n\nConversation ended by user. Here are your insights:")
            print(conversational_ai.get_conversation_insights())
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again...")
            continue
if __name__ == "__main__":
    main()   
