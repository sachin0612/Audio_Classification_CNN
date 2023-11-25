import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import librosa
import cv2

# Load your trained model
model_path = 'audio_classification.h5'
model = load_model(model_path)

# Classes
y=['dog_bark', 'children_playing', 'car_horn', 'air_conditioner',
       'street_music', 'gun_shot', 'siren', 'engine_idling', 'jackhammer',
       'drilling']
le.fit(y)

# Function to preprocess the uploaded audio file
def features_extract(file):
    sample,sample_rate = librosa.load(file)
    feature = librosa.feature.mfcc(y=sample,sr=sample_rate,n_mfcc=40)
    up_points = (173,40)
    scaled_feature= cv2.resize(feature, up_points, interpolation= cv2.INTER_LINEAR)
    scaled_feature=scaled_feature.reshape(1,40,173,1)
    return scaled_feature

# Function to make a prediction using the loaded model
def print_prediction(file_name):
    pred_fea = features_extract(file_name)
    pred_vector = np.argmax(model.predict(pred_fea), axis=-1)
    pred_class = le.inverse_transform(pred_vector)
    return pred_class[0]

# Streamlit app
st.title("Sound Classification App")
st.write("Upload an audio file, and the model will predict the sound class.")
st.write("This app only classify 10 objects sounds which are: dog_bark , children_playing, car_horn, air_conditioner, street_music, gun_shot, siren, engine_idling, jackhammer, drilling")

# Sidebar with link to resource
st.sidebar.markdown("# Concept Resource")
st.sidebar.markdown("[Read about Sound Classification using deep Learning](https://towardsdatascience.com/audio-deep-learning-made-simple-part-1-state-of-the-art-techniques-da1d3dff2504)")

st.sidebar.markdown("# GitHub Link")
st.sidebar.markdown("[Click here !](https://github.com/sachin0612/Audio_Classification_CNN)")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav", start_time=0)

    if st.button("Predict"):
        # Make prediction
        prediction_class = print_prediction(uploaded_file)
        # Map the predicted class index to the class name
        

        st.success(f"Prediction: {prediction_class}")
