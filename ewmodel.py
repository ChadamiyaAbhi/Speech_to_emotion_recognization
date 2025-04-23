import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('final_emotion_recognition_model.h5')

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # Update if necessary

def predict_emotion(audio_path):
    """
    Predict the emotion from an audio file.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        str: Predicted emotion label.
    """
    try:
        # Load and preprocess the audio file
        audio, sr = librosa.load(audio_path, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        features = np.mean(mfcc.T, axis=0)

        # Reshape features to match model input
        test_features = np.expand_dims(features, axis=0)
        test_features = np.expand_dims(test_features, axis=-1)

        # Make predictions
        predictions = model.predict(test_features)
        predicted_class = np.argmax(predictions)
        
        # Return the emotion label
        return emotion_labels[predicted_class]

    except Exception as e:
        return f"Error processing audio file: {e}"

# Main execution
if __name__ == "__main__":
    # Specify the path to the audio file you want to test
    audio_path = 'YAF_wheat_happy.wav'  # Replace this with your audio file path
    
    # Predict emotion
    predicted_emotion = predict_emotion(audio_path)
    
    # Display the result
    print(f"Predicted Emotion: {predicted_emotion}")
