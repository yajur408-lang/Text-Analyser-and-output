import streamlit as st
import sklearn
import random
import os
from PIL import Image
import pytesseract
from pathlib import Path
st.title("Teen persona Assistant")
# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# Open an image file
path=Path("C:/Users/Nivesh Ahluwalia/Downloads/test1.png")
img = Image.open(path)

text = pytesseract.image_to_string(img)

teen_dictionary = {
    "SMH": {"meaning": "Shaking my head", "example": "SMH, he still hasn’t replied", "tone": "Annoyed / Frustrated"},
    "LOL": {"meaning": "Laughing out loud", "example": "That meme was so funny LOL", "tone": "Funny / Casual"},
    "LMAO": {"meaning": "Laughing my ass off", "example": "LMAO you’re crazy", "tone": "Hype / Funny"},
    "ROFL": {"meaning": "Rolling on the floor laughing", "example": "ROFL this is too much", "tone": "Crazy / Funny"},
    "BRB": {"meaning": "Be right back", "example": "BRB, grabbing food", "tone": "Casual"},
    "BFF": {"meaning": "Best friends forever", "example": "You’re my BFF", "tone": "Supportive / Positive"},
    "IDK": {"meaning": "I don’t know", "example": "IDK what he meant", "tone": "Neutral / Casual"},
    "ILY": {"meaning": "I love you", "example": "ILY so much 😘", "tone": "Affectionate"},
    "IMHO": {"meaning": "In my humble opinion", "example": "IMHO that movie was trash", "tone": "Opinionated / Casual"},
    "TBH": {"meaning": "To be honest", "example": "TBH I didn’t like it", "tone": "Honest / Casual"},
    "TTYL": {"meaning": "Talk to you later", "example": "GTG, TTYL", "tone": "Casual"},
    "GTG": {"meaning": "Got to go", "example": "GTG to class", "tone": "Casual"},
    "FOMO": {"meaning": "Fear of missing out", "example": "I have FOMO rn", "tone": "Anxious / Casual"},
    "YOLO": {"meaning": "You only live once", "example": "YOLO let’s go", "tone": "Excited / Bold"},
    "TFW": {"meaning": "That feeling when", "example": "TFW exam is tomorrow", "tone": "Relatable / Casual"},
    "ICYMI": {"meaning": "In case you missed it", "example": "ICYMI here’s the link", "tone": "Informative / Casual"},
    "NSFW": {"meaning": "Not safe for work", "example": "This link is NSFW", "tone": "Warning / Serious"},
    "NSFL": {"meaning": "Not safe for life", "example": "That video is NSFL", "tone": "Shocked / Serious"},
    "FML": {"meaning": "F*** my life", "example": "FML I lost my phone", "tone": "Frustrated"},
    "BTW": {"meaning": "By the way", "example": "BTW that was wild", "tone": "Casual"},
    "HMU": {"meaning": "Hit me up", "example": "HMU later", "tone": "Casual"},
    "Bae": {"meaning": "Before anyone else", "example": "Going out with bae", "tone": "Affectionate"},
    "IDC": {"meaning": "I don’t care", "example": "IDC what they think", "tone": "Dismissive / Chill"},
    "ILYSM": {"meaning": "I love you so much", "example": "ILYSM ❤️", "tone": "Affectionate"},
    "IC": {"meaning": "I see", "example": "Oh, IC now", "tone": "Casual"},
    "TMI": {"meaning": "Too much information", "example": "TMI dude", "tone": "Annoyed"},
    "WYD": {"meaning": "What are you doing?", "example": "WYD rn?", "tone": "Casual"},
    "WDYM": {"meaning": "What do you mean?", "example": "WDYM that’s wrong?", "tone": "Confused / Casual"},
    "NVM": {"meaning": "Never mind", "example": "NVM it’s fine", "tone": "Dismissive"},
    "OMW": {"meaning": "On my way", "example": "OMW, be there soon", "tone": "Casual"},
    "PFP": {"meaning": "Profile picture", "example": "Change your PFP", "tone": "Casual"},
    "GG": {"meaning": "Good game", "example": "GG everyone", "tone": "Respectful / Casual"},
    "GLHF": {"meaning": "Good luck, have fun", "example": "GLHF in the match", "tone": "Supportive"},
    "AFK": {"meaning": "Away from keyboard", "example": "AFK for lunch", "tone": "Casual"},
    "FTW": {"meaning": "For the win", "example": "That move FTW", "tone": "Excited"},
    "IKR": {"meaning": "I know, right?", "example": "IKR that was insane", "tone": "Agreeing"},
    "JK": {"meaning": "Just kidding", "example": "JK, relax 😂", "tone": "Playful"},
    "ILY2": {"meaning": "I love you too", "example": "ILY2 ❤️", "tone": "Affectionate"},
    "OTP": {"meaning": "One true pairing", "example": "They’re my OTP", "tone": "Fan / Positive"},
    "LYLAS": {"meaning": "Love you like a sister", "example": "LYLAS", "tone": "Affectionate"},
    "MFW": {"meaning": "My face when", "example": "MFW we won", "tone": "Relatable / Funny"},
    "BRUH": {"meaning": "Seriously?", "example": "Bruh, really?", "tone": "Annoyed / Casual"},
    "SUS": {"meaning": "Suspicious", "example": "That seems SUS", "tone": "Skeptical"},
    "GOAT": {"meaning": "Greatest of all time", "example": "He’s the GOAT", "tone": "Admiring"},
    "FYP": {"meaning": "For You Page", "example": "This blew up on my FYP", "tone": "Excited"},
    "WTF": {"meaning": "What the f***", "example": "WTF happened?", "tone": "Shocked / Angry"},
    "ILYSM": {"meaning": "I love you so much", "example": "ILYSM", "tone": "Affectionate"},
    "ILYSMF": {"meaning": "I love you so much fr", "example": "ILYSMF ❤️", "tone": "Affectionate"},
    "TBHIDK": {"meaning": "To be honest I don’t know", "example": "TBHIDK", "tone": "Honest / Casual"},
    "YAAAS": {"meaning": "Yes!", "example": "YAAAS queen!", "tone": "Excited"},
    "SAV": {"meaning": "Savage", "example": "That was SAV", "tone": "Bold / Playful"},
    "RN": {"meaning": "Right now", "example": "I’m busy RN", "tone": "Casual"},
    "ILY4E": {"meaning": "I love you forever", "example": "ILY4E ❤️", "tone": "Affectionate"},
    "HBU": {"meaning": "How about you?", "example": "I’m good, HBU?", "tone": "Casual"},
    "ILY4EV": {"meaning": "I love you forever", "example": "ILY4EV", "tone": "Affectionate"},
    "FTL": {"meaning": "For the loss (sarcastic)", "example": "FTL we lost again", "tone": "Sarcastic"},
    "GTFO": {"meaning": "Get the f*** out", "example": "GTFO lol", "tone": "Playful / Strong"},
    "IDCLOL": {"meaning": "I don’t care lol", "example": "IDCLOL", "tone": "Casual"},
    "ILYSMT": {"meaning": "I love you so much too", "example": "ILYSMT", "tone": "Affectionate"},
    "ICYDK": {"meaning": "In case you didnt know", "example": "ICYDK this is trending", "tone": "Informative"},
    "HBD": {"meaning": "Happy birthday", "example": "HBD!!!", "tone": "Celebratory"},
    "GN": {"meaning": "Good night", "example": "GN everyone", "tone": "Warm / Casual"},
    "GM": {"meaning": "Good morning", "example": "GM ☀️", "tone": "Warm / Casual"},
    "ILYSMBF": {"meaning": "I love you so much best friend", "example": "ILYSMBF", "tone": "Affectionate"},
    "POS": {"meaning": "Parent over shoulder", "example": "POS rn, btw", "tone": "Cautious"},
    "TBF": {"meaning": "To be fair", "example": "TBF that was okay", "tone": "Balanced"},
    
}
from google import genai
API_KEY = "AIzaSyBdwZKn4TF5AKuqfNpFUJlVAVkDXdB-RBI"  # replace with your actual API key
MODEL = "gemini-2.5-flash"  # valid Gemini model
client = genai.Client(api_key=API_KEY)
def ask_gemini():
    try:
        prompt = (
            "You are a helpful assistant which helps people respond to certain scenarios and changes the tone according to the severity of the scenario. "
            f"Text: {text}\n Provide a possible response to this scenario. Use slangs and acronyms where necessary. "
            "You can refer to the teen_dictionary for guidance if needed."
        )

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        answer = response.text.strip()
        if not answer:
            st.write(f'"{text}" is not recognized as a common teen slang or acronym.')
        else:
            st.write(f"Answer to your question: {answer}")

    except Exception as e:
        st.error(f"Server down or API error: {e}")
ask_gemini()
