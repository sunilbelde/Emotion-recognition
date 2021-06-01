import streamlit as st
import numpy as np    
import tensorflow as tf
import librosa # to extract speech features
import pyaudio
import wave


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 3.5
WAVE_OUTPUT_FILENAME = "input.wav"

p = pyaudio.PyAudio()


def record_audio():
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
def main():
    #print(cv2.__version__)
    selected_box = st.sidebar.selectbox(
        'Choose an option..',
        ('Emotion Recognition','hello')
        )
            
    if selected_box == 'Emotion Recognition':        
        st.sidebar.success('To try by yourself select "Evaluate the model".')
        application()

    
@st.cache(show_spinner=False)
def load_model():
    model=tf.keras.models.load_model('mymodel.h5')
    
    return model
def application():
    models_load_state=st.text('\n Loading models..')
    model=load_model()
    models_load_state.text('\n Models Loading..complete')
    
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording
    
    record = st.button('Record Now')
    
    if record:
        record_audio()
        st.success('Emotion of the audio is  ',predict(model,'input.wav'))

def extract_mfcc(wav_file_name):
    #This function extracts mfcc features and obtain the mean of each dimension
    #Input : path_to_wav_file
    #Output: mfcc_features'''
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    
    return mfccs
    
    
def predict(model,wav_filepath):
    emotions={1 : 'neutral', 2 : 'calm', 3 : 'happy', 4 : 'sad', 5 : 'angry', 6 : 'fearful', 7 : 'disgust', 8 : 'surprised'}
    test_point=extract_mfcc(wav_filepath)
    test_point=np.reshape(test_point,newshape=(1,40,1))
    predictions=model.predict(test_point)
    print(emotions[np.argmax(predictions[0])+1])
    
    return emotions[np.argmax(predictions[0])+1]
if __name__ == "__main__":
    main()
