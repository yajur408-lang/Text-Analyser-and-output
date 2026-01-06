import streamlit as st
import google.generativeai as genai

# 1. SETUP - Replace the string below with your actual API key
API_KEY = "AIzaSyDQRRS0qCCzfYaYEq7nZUliBwkA62FmqL8" 

# Configure the library
try:
    genai.configure(api_key=API_KEY)
    # Check if the key works by initializing the model
    model = genai.GenerativeModel("gemini-2.5-flash")
except Exception as e:
    st.error(f"Configuration Error: {e}")

# 2. UI LAYOUT
st.set_page_config(page_title="Email Upgrader", page_icon="✉️")
st.title("✉️ Email Upgrader")
st.write("Convert your drafts into professional, native-level English.")

# Input area
email_text = st.text_area("Paste your email here:", height=200, placeholder="Write your draft...")

# Options
tone = st.selectbox("Select desired tone:", ["Professional", "Formal", "Friendly", "Polite"])

# 3. LOGIC
if st.button("Upgrade Email", type="primary"):
    if not email_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Refining your email..."):
            try:
                # Direct prompt construction
                prompt = (
                    f"Rewrite this email to sound like a native English speaker. "
                    f"Tone: {tone}. "
                    f"Keep the exact same meaning, but improve grammar and flow. "
                    f"Return ONLY the rewritten email text.\n\n"
                    f"Original Email:\n{email_text}"
                )
                
                # Call the API
                response = model.generate_content(prompt)
                
                # Display Results
                st.subheader("Improved Version:")
                st.info("You can copy the text from the box below:")
                st.code(response.text, language="text")
                
            except Exception as e:
                # Specific error handling for invalid keys
                if "API_KEY_INVALID" in str(e) or "400" in str(e):
                    st.error("❌ Invalid API Key. Please check your key in Google AI Studio and ensure it is copied correctly.")
                else:
                    st.error(f"An error occurred: {e}")

# Footer note
st.divider()
st.caption("Tip: If you get a 400 error, wait 60 seconds for a new key to activate.")
