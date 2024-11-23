import streamlit as st
from PIL import Image
import pytesseract
import pyttsx3
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize Google Generative AI with API Key
GEMINI_API_KEY = "(your api key)"  # Replace with your valid API key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=GEMINI_API_KEY)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Set up Streamlit page with custom configurations
st.set_page_config(page_title="OptiAssist",initial_sidebar_state="expanded")

# Set title and sidebar
st.title("AI Assistant for Visually Impaired")
st.sidebar.title("🔧 Features")
st.sidebar.markdown("""
Welcome to **OPTIASSIST**! This app helps visually impaired users interact with images in meaningful ways.
- **Scene Understanding**: Describes the content of the uploaded image.
- **Text-to-Speech**: Converts the scene description or extracted text in images to speech.
- **Object & Obstacle Detection**: Detects objects and potential obstacles.
- **Personalized Assistance**: Provides task-specific guidance based on the uploaded image.
""")
st.sidebar.markdown("### How to Use")
st.sidebar.markdown("""
1. Upload an image in the main section above.
2. Choose a feature to interact with, such as **Describe Scene**, **Text-to-Speech**, or **Personalized Assistance**.
3. The app will process the image and provide relevant guidance or output with an option to hear it aloud.
""")

# Functions for processing
def generate_scene_description(input_prompt, image_data):
    """Generates a scene description using Google Generative AI."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

def generate_task_guidance(input_prompt, image_data):
    """Generates task-specific guidance based on the image."""
    # You can add custom prompts based on recognized items, labels, or objects
    task_prompt = f"""
    You are an AI assistant helping visually impaired individuals by recognizing objects and providing task-specific guidance.
    Based on the image, identify useful items (e.g., food, clothing, products) and provide suggestions for daily tasks.
    Provide instructions on how to interact with recognized items, read labels, or identify objects for tasks such as cooking, dressing, or other activities.
    """
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([task_prompt, image_data[0]])
    return response.text

def input_image_setup(uploaded_file):
    """Prepares the uploaded image for processing."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded.")

# Main app functionality
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

# Layout: two columns for buttons and image
col1, col2 = st.columns([1, 3])  # Button section on the left (col1), image on the right (col2)

with col1:
    # Make all buttons the same size by using 'use_container_width=True'
    scene_button = st.button("🔍 Describe Scene", use_container_width=True)
    tts_button = st.button("🔊 Text-to-Speech", use_container_width=True)
    task_button = st.button("📝 Personalized Assistance", use_container_width=True)

with col2:
    # Display the uploaded image if it exists, resized to a smaller size
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=500)  # Resize image to 500px wide, further to the right

# Input Prompt for AI Scene Understanding
input_prompt = """
You are an AI assistant helping visually impaired individuals by describing the scene in the image. Provide:
1. List of items detected in the image with their purpose.
2. Overall description of the image.
3. Suggestions for actions or precautions for the visually impaired.
"""

# Process based on user interaction
if uploaded_file:
    image_data = input_image_setup(uploaded_file)

    if scene_button:
        with st.spinner("Generating scene description..."):
            response = generate_scene_description(input_prompt, image_data)
            st.subheader("Scene Description")
            st.write(response)

            # Convert the scene description to speech
            st.write("🔊 Audio Output:")
            engine.save_to_file(response, 'scene_description.mp3')
            engine.runAndWait()

            # Play audio
            audio_file = open('scene_description.mp3', 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
            st.success("Scene description read aloud!")

    if tts_button:
        with st.spinner("Converting scene description to speech..."):
            # Generate the scene description for Text-to-Speech conversion
            scene_description = generate_scene_description(input_prompt, image_data)

            # Show the generated scene description as text
            st.subheader("Scene Description ✍️:")
            st.text(scene_description)

            # Convert text to speech
            st.write("🔊 Audio Output:")
            engine.save_to_file(scene_description, 'scene_description.mp3')
            engine.runAndWait()

            # Play audio
            audio_file = open('scene_description.mp3', 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
            st.success("Text-to-Speech Conversion Completed!")

    if task_button:
        with st.spinner("Generating personalized assistance..."):
            # Generate task-specific guidance based on the uploaded image
            task_guidance = generate_task_guidance(input_prompt, image_data)

            # Show the generated task guidance as text
            st.subheader("Personalized Assistance ✍️:")
            st.text(task_guidance)

            # Convert task guidance to speech
            st.write("🔊 Audio Output:")
            engine.save_to_file(task_guidance, 'task_guidance.mp3')
            engine.runAndWait()

            # Play audio
            audio_file = open('task_guidance.mp3', 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
            st.success("Task guidance read aloud!")
