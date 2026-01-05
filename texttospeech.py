import pyttsx3
import streamlit as st
engine = pyttsx3.init()
st.title("Text to Speech Converter")
# Set properties (optional)
rate=120 # Speed percent (can go over 100)
engine.setProperty('rate', rate)  # Speed (words per minute)
evolume=0.7 # Volume (0.0 to 1.0)
engine.setProperty('volume', evolume)  #set according to https://www.atia.org/wp-content/uploads/2024/04/ATOB_V18.pdf?utm_source=chatgpt.com


# Input text
text = st.text_input("Enter text to speak: ")
stop=st.button("Stop Speaking")
if stop:
    engine.stop()
# Speak
engine.say(text)
engine.runAndWait()
