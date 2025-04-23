# import os
# import librosa
# import numpy as np
# import soundfile as sf
# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# # Set upload folder
# UPLOAD_FOLDER = "uploads"
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# # Allowed file extensions
# ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg"}

# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# # Dummy Emotion Prediction Function (Replace with ML Model)
# def predict_emotion(audio_path):
#     try:
#         y, sr = librosa.load(audio_path, sr=None)
#         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#         features = np.mean(mfccs, axis=1)

#         # Simulating emotion detection
#         emotions = ["Happy", "Sad", "Angry", "Neutral"]
#         return np.random.choice(emotions)  # Replace with actual ML model
#     except Exception as e:
#         return str(e)

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/upload", methods=["POST"])
# def upload_audio():
#     if "file" not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files["file"]

#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#         file.save(filepath)

#         # Predict emotion
#         emotion = predict_emotion(filepath)

#         return jsonify({"message": "File processed", "emotion": emotion}), 200

#     return jsonify({"error": "Invalid file type"}), 400

# if __name__ == "__main__":
#     app.run(debug=True)



























import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import sounddevice as sd
import wavio

# Load the trained emotion recognition model
model = load_model('emotion_recognition_model2.h5')

# Define emotion labels (make sure they match your model training)
emotion_labels = ['happy', 'sad', 'angry', 'neutral']

# Function to convert audio files to WAV format
def convert_to_wav(input_file, output_file):
    try:
        print(f"Converting {input_file} to WAV format...")
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        print(f"File converted and saved as {output_file}")
    except Exception as e:
        print(f"Error converting file: {e}")

# Function to preprocess audio for emotion recognition
def preprocess_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)  # Load audio
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # Extract MFCC features
        features = np.mean(mfcc.T, axis=0)  # Aggregate MFCCs
        return np.expand_dims(features, axis=0)  # Reshape for model input
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# Function to predict emotion from audio
def predict_emotion(file_path):
    print(f"Processing file: {file_path}")
    features = preprocess_audio(file_path)
    if features is not None:
        predictions = model.predict(features)
        predicted_class = np.argmax(predictions)
        print(f"Predicted Emotion: {emotion_labels[predicted_class]}")
        return emotion_labels[predicted_class]
    else:
        print("Could not process the audio file.")
        return None

# Function to record audio from microphone
def record_audio(output_file, duration=5, sample_rate=44100):
    print(f"Recording audio for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    wavio.write(output_file, recording, sample_rate, sampwidth=2)
    print(f"Audio recorded and saved as {output_file}")

# Main function
if __name__ == "__main__":
    print("=== Emotion Recognition System ===")
    print("1. Analyze existing audio file")
    print("2. Record and analyze audio")
    choice = input("Enter your choice (1/2): ").strip()

    if choice == "1":
        input_file = input("Enter the path of the audio file (M4A, MP4, WAV): ").strip()
        if not os.path.exists(input_file):
            print("Error: File not found.")
        else:
            if input_file.endswith(".wav"):
                predict_emotion(input_file)
            else:
                wav_file = "converted_audio.wav"
                convert_to_wav(input_file, wav_file)
                predict_emotion(wav_file)

    elif choice == "2":
        output_file = "recorded_audio.wav"
        duration = int(input("Enter recording duration in seconds: ").strip())
        record_audio(output_file, duration)
        predict_emotion(output_file)

    else:
        print("Invalid choice. Exiting.")