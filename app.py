import streamlit as st
import requests
import os
from io import BytesIO
from typing import List
import sounddevice as sd
import numpy as np
import wave
from datetime import datetime
from huggingface_hub import InferenceClient

# Configuration
API_URL = "http://localhost:8000"  # Update if your API is hosted elsewhere
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state
if "documents" not in st.session_state:
    st.session_state.documents = []
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_frames" not in st.session_state:
    st.session_state.audio_frames = []

# Initialize Whisper client (for speech-to-text)
try:
    whisper_client = InferenceClient(model="openai/whisper-base")
except Exception as e:
    st.warning(f"Could not initialize Whisper client: {e}")
    whisper_client = None

# UI Layout
st.set_page_config(page_title="AI Slide Generator", layout="wide")
st.title("AI-Powered Slide Generator with RAG")

# Sidebar for document management
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader(
        "Upload documents for RAG (PDF, DOCX, TXT, MD)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in [d["name"] for d in st.session_state.documents]:
                try:
                    # Save file locally
                    file_path = os.path.join(UPLOAD_DIR, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Upload to API
                    files = {"file": (file.name, file.getvalue(), file.type)}
                    response = requests.post(f"{API_URL}/upload-document/", files=files)
                    
                    if response.status_code == 200:
                        doc_data = response.json()
                        st.session_state.documents.append({
                            "id": doc_data["document_id"],
                            "name": file.name,
                            "status": "uploaded"
                        })
                        st.success(f"Uploaded {file.name}")
                    else:
                        st.error(f"Failed to upload {file.name}")
                except Exception as e:
                    st.error(f"Error uploading {file.name}: {str(e)}")
    
    if st.session_state.documents:
        st.subheader("Uploaded Documents")
        for doc in st.session_state.documents:
            st.info(f"{doc['name']} - {doc['status']}")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Presentation Generator")
    
    # Speech-to-text section
    st.write("### Voice Input")
    if st.button("🎤 Start Recording", key="record_button"):
        st.session_state.recording = True
        st.session_state.audio_frames = []
        st.experimental_rerun()
    
    if st.button("⏹ Stop Recording", key="stop_button") and st.session_state.recording:
        st.session_state.recording = False
        st.experimental_rerun()
    
    if st.session_state.recording:
        st.warning("Recording in progress... Speak now!")
        
        # Record audio
        samplerate = 16000
        duration = 10  # seconds
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
        sd.wait()
        st.session_state.audio_frames = recording
        
        # Convert to bytes
        audio_bytes = BytesIO()
        with wave.open(audio_bytes, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes((st.session_state.audio_frames * 32767).astype(np.int16))
        audio_bytes.seek(0)
        
        # Transcribe
        if whisper_client:
            try:
                transcription = whisper_client.automatic_speech_recognition(audio_bytes)
                st.session_state.prompt = transcription
                st.success(f"Transcription: {transcription}")
            except Exception as e:
                st.error(f"Transcription failed: {str(e)}")
    
    # Text input
    prompt = st.text_area(
        "Enter your presentation topic or instructions:",
        value=st.session_state.get("prompt", ""),
        height=150,
    )
    
    # Presentation options
    with st.expander("Advanced Options"):
        col3, col4 = st.columns(2)
        with col3:
            slide_count = st.slider("Number of slides", 1, 20, 5)
            style = st.selectbox(
                "Presentation style",
                ["Professional", "Academic", "Creative", "Minimalist"],
            )
        with col4:
            selected_docs = st.multiselect(
                "Use documents for context",
                [d["name"] for d in st.session_state.documents],
            )
            generate_images = st.checkbox("Generate images", value=False)
    
    # Generate button
    if st.button("Generate Presentation", type="primary"):
        if not prompt:
            st.error("Please enter a prompt or use voice input")
        else:
            with st.spinner("Generating your presentation..."):
                try:
                    # Map selected doc names to their IDs
                    doc_ids = [
                        d["id"] for d in st.session_state.documents 
                        if d["name"] in selected_docs
                    ]
                    
                    # Prepare request
                    data = {
                        "prompt": prompt,
                        "document_ids": doc_ids,
                        "generate_images": generate_images,
                        "slide_count": slide_count,
                        "style": style.lower(),
                    }
                    
                    # Call API
                    response = requests.post(f"{API_URL}/generate-slides/", json=data)
                    
                    if response.status_code == 200:
                        # Save the PPTX
                        pptx_bytes = BytesIO(response.content)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"presentation_{timestamp}.pptx"
                        
                        st.success("Presentation generated successfully!")
                        st.download_button(
                            label="Download Presentation",
                            data=pptx_bytes,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        )
                    else:
                        st.error(f"Generation failed: {response.text}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

with col2:
    st.subheader("Instructions")
    st.markdown("""
    1. **Upload documents** (optional) in the sidebar
    2. **Enter your topic** or use voice input
    3. **Select options** (slides count, style)
    4. **Generate** your presentation
    
    **Features:**
    - RAG from uploaded documents
    - AI-generated content
    - Optional AI images
    - Voice commands
    """)
    
    st.subheader("Example Prompts")
    st.markdown("""
    - "Create a 5-slide intro to machine learning"
    - "Make a presentation about climate change using my uploaded reports"
    - "Generate slides about AI ethics with 3 bullet points per slide"
    """)

# Hide Streamlit branding
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)