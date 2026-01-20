import streamlit as st
import google.generativeai as genai

# 1. SETUP - Replace the string below with your actual API key
API_KEY = "" 

# Configure the library
try:
    genai.configure(api_key=API_KEY)
    # Check if the key works by initializing the model
    model = genai.GenerativeModel("gemini-2.5-flash")
except Exception as e:
    st.error(f"Configuration Error: {e}")

# 2. UI LAYOUT
st.set_page_config(page_title="Email Upgrader", page_icon="✉️")
st.title("✉️ Improve Your Email/")
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
                    f"Format of letter writing:"
                    f"1. Use proper salutation (e.g., 'Dear [Name],')"
                    f"2. Include a clear subject line"
                    f"3. Structure the body with clear paragraphs"
                    f"4. End with a professional closing (e.g., 'Best regards,')"
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
import os
import requests

# Load client credentials from environment variables
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
TOKEN_ENDPOINT = "https://auth.grammarly.com/v4/api/oauth2/token"
SCOPE = "users-api:read"
API_ENDPOINT = "https://api.grammarly.com/ecosystem/api/institutions-summary"

# Step 1: Request Access Token
def get_access_token():
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": SCOPE
    }
    response = requests.post(TOKEN_ENDPOINT, data=payload)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json().get("access_token")

# Step 2: Make Authenticated API Request
def fetch_user_data():
    access_token = get_access_token()
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(API_ENDPOINT, headers=headers)
    response.raise_for_status()
    return response.json()

# Example usage
if __name__ == "__main__":
    data = fetch_user_data()
    print(data)
