# Multi-Modal Chatbot Application Project

This document outlines the documentation, setup and explanation of the code used for development of the Multi-Modal Chatbot application project. The application is implemented in **Python** which integrates text and image generation functionalities using **Streamlit**, **Google Generative AI**, and **Pollinations AI**. It supports generating AI-based responses and images based on user input.

---

## Overview

Multi-Model chatbot is a chatbot application that combines text and image generation capabilities. Not only that it also understands images and can respond to them, it can also generate images based on user prompts or variations of previously generated images. Hence making it a versatile and interactive chatbot.

To implement this multi-model functionality, the application uses the following technique:

1. **Text Generation**: Utilizes Google Generative AI's `gemini-1.5-flash` model for text generation.
2. **Text History**: Utilizes the Google Generative AI's `start_chat()` method to start a conversation with the model while storing the chat history.
3. **Image Generation**: Utilizes Pollinations AI's `flux` model for image generation.
4. **Image Understanding**: Utilizes Google Generative AI's `gemini-1.5-flash` model existing features for image understanding.
5. **Image Variation**: Utilizes Google Generative AI's `gemini-1.5-flash` model to understand when the user requests image variation, and then prompts Pollinations AI's `flux` model for image variation.
6. **Contextual Integrity**: Utilizes multiple Google Generative AI's `gemini-1.5-flash` models to always keep a complete contextual understanding of the entire conversation at all times for both text and image.

## Prerequisites

### Libraries and Tools

- Python 3.12
- Required Python libraries:
  - `os`
  - `pillow`
  - `streamlit`
  - `pollinations`
  - `random`
  - `datetime`
  - `dotenv`
  - `google.generativeai`
- `GEMINI_API_KEY`: API key for Google Generative AI.

### Installation and Setup

1. Setup a virtual environment

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install the required libraries:

   ```bash
   pip install pillow streamlit pollinations python-dotenv google-generativeai
   ```

3. Environment Variables:
   - Set up an `.env` file with the following key:

    ```env
    GEMINI_API_KEY=<Your Google Generative AI API Key>
    ```

### Usage

1. Run the Streamlit application:

    ```bash
    streamlit run task2_multi_modal_chatbot\src\app.py
    ```

2. Open the application in your web browser:

   - Navigate to `http://localhost:8501` in your web browser.

3. Use the application to generate text and images based on user prompts:

   - Enter text prompts in the text input field.
       - eg. `Hello, how are you?`
   - Use `/generate {prompt}` to request an image.
       - eg. `/generate a futuristic cityscape at night` to generate images.
       - eg. `/generate a variation of the last generated image` to generate variations of the last generated image.
   - You can Upload the images with the `Browse File` Button/ Container or just drag and drop an Image on the Button.
   - The uploaded images and generated images will be displayed in the chat interface and also stored locally for contextual understanding for the model.

---

## Files

- **`app.py`**: Main Streamlit application file. (Everything is contained in this file)

## Implementation

- **Custom CSS Styling**:
  - Provides a customized user interface with modified layouts for chat and file upload components.
  - Modifies the layout of Streamlit components for better usability.
  - Hides unnecessary elements like default file uploader labels.

- **Model Initialization**:
  - Configures Google Generative AI models for both text and image generation.
  - Pollinations AI is used for generating images with specific parameters (e.g., resolution, seed).
  - Initializes text and image generation models:
    - **Text Model**: `gemini-1.5-flash`
    - **Image Model**: `flux` (Pollinations AI).

- **Text Generation/ Chat Functionality**:
  - Stores chat history in the Streamlit session state.
  - Initializes chat history and image-related session states in Streamlit.
  - Initializes text, image and context models in the session state.

- **Image Upload and Generation**:
  - Allows users to upload images or request image generation.
  - Generates images based on user prompts or variations of previously generated images.
  - Displays uploaded and generated images in the chat interface.
  - Uploaded and generated images are saved locally with timestamped filenames for easy identification.

- **Prompt Processing and Contextual Understanding**:
  - Maintains a contextual understanding of the entire conversation.
  - Prompts itself and other models to understand the context of the conversation.
  - Detects if the user input requests image generation or variations of existing images and prompts the appropriate model.
  - Generates appropriate prompts for image generation or response generation.

- **Error Handling and Logs**:
  - Handles API errors and other exceptions gracefully, displaying error messages to the user.
  - Logs key events like model initialization, image saving, and prompt processing to the console for debugging purposes.

---

## Conclusion

The Multi-Modal Chatbot combines advanced AI capabilities to deliver an engaging and interactive user experience with text and image generation capabilities.
