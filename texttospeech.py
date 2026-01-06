import streamlit as st
from gtts import gTTS
from io import BytesIO

st.title("Text-to-Speech Converter")

# Text input
text = st.text_area("Enter text to convert to speech:")

# Button to generate speech
if st.button("Play Speech"):
    if text.strip() != "":
        # Convert text to speech and store in memory
        tts = gTTS(text=text, lang='en', slow=False)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)  # Move to the start of the BytesIO buffer
        
        # Play the audio in Streamlit
        st.audio(audio_bytes, format='audio/mp3')
    else:
        st.error("Please enter some text to convert.")

c
