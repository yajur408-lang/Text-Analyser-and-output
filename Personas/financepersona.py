import streamlit as st
from PIL import Image
import pytesseract
import google.generativeai as genai

# Streamlit title
st.title("Professional in Finance Persona Assistant")

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# File uploader
uploaded_file = st.file_uploader("Upload an image containing text", type=["png", "jpg", "jpeg"])

# Gemini API setup
API_KEY = "YOUR_API_KEY"  # ⚠️ Replace with environment variable for safety
genai.configure(api_key=API_KEY)
MODEL = "gemini-2.5-flash"

if uploaded_file is not None:
    # Open the uploaded image
    img = Image.open(uploaded_file)
    
    # Extract text from the image
    text = pytesseract.image_to_string(img).strip()
    
    if text:
        st.write("**Extracted text:**")
        st.write(text)

        # Generate professional finance response
        with st.spinner("Generating professional finance response..."):
            try:
                prompt = (
                    ": "
                    f"{text}"
                )

                response = genai.generate(
                    model=MODEL,
                    prompt=prompt
                )

                answer = response.text.strip()
                st.write("**Finance Expert Response:**")
                st.write(answer)

            except Exception as e:
                st.error(f"Server down or API error: {e}")
    else:
        st.warning("No text could be extracted from the image. Try a clearer image.")
