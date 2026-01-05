import google.generativeai as genai
import streamlit as st
# ---------------- CONFIG ----------------
API_KEY = "AIzaSyAwL5_A9QyZ-sqYsuhFvDhm1908ANYNH9I"  #Replace with your api key I dont have many credits
MODEL_NAME = "gemini 2.0 Flash-Lite"
# ----------------------------------------
st.title("Acroynm and tone explainer")
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(MODEL_NAME)
def ask_gemini():
    question = st.text_input("Enter your question:")
    

    try:
        prompt = (
            "You are a precise assistant. Always provide a single, definitive answer and explain briefly your reasoning. "
            f"text\n{question}\n"+"What does it mean as an acronym?"
        )
        
        response = model.generate_content(prompt)
        st.write(f"Q: {question}")
        st.write(f"A: {response.text}")
    except Exception as e:
        st.error(f"Server down: {e}")
ask_gemini()
