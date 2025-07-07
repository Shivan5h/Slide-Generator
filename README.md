# AI-Powered Slide Generator with RAG

This project is a Streamlit web application that generates presentation slides using AI, with support for Retrieval-Augmented Generation (RAG) from your uploaded documents. It also features voice input for prompts and optional AI-generated images.

## Features

- **Upload Documents:** Supports PDF, DOCX, TXT, and Markdown files for RAG context.
- **Voice Input:** Use your microphone to dictate your presentation topic or instructions.
- **AI Slide Generation:** Generates PowerPoint presentations based on your prompt and selected options.
- **Image Generation:** Optionally generate AI images for your slides.
- **Downloadable Output:** Download your generated presentation as a PPTX file.

## Getting Started

### Prerequisites
- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)

### Installation
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
1. Start the backend API (make sure it is running at the URL specified in `API_URL` in `app.py`).
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser to the provided local URL.

## Usage
1. **Upload documents** (optional) in the sidebar.
2. **Enter your topic** or use the voice input feature.
3. **Select options** (number of slides, style, use images, etc.).
4. **Generate** your presentation and download the PPTX file.

## Configuration
- The backend API URL can be changed by modifying the `API_URL` variable in `app.py`.
- Uploaded files are stored in the `uploads/` directory.

## Notes
- The app uses Hugging Face's Whisper model for speech-to-text.
- Make sure your microphone is enabled for voice input.
- The backend API must implement `/upload-document/` and `/generate-slides/` endpoints.
