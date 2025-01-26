# Multi-Modal Sentimental Chatbot Application Project

This document outlines the documentation, setup and explanation of the code used for development of the Multi-Modal Sentimental Chatbot application project. The application is implemented in **Python** which integrates text and image generation functionalities using **Streamlit**, **Google Generative AI**, and **Pollinations AI**. The application also uses a custom fine-tuned model for sentiment analysis which was fine-tuned using a public dataset of sentiment analysis and a pre-trained model using **Google Generative AI**. It also supports all the multi-model capabilities and is built on top of the Multi-Modal Chatbot application project.

---

## Overview

Multi-Model chatbot is a chatbot application that combines text and image generation capabilities. Not only that it also understands images and can respond to them, it can also generate images based on user prompts or variations of previously generated images. Hence making it a versatile and interactive chatbot.

To implement this multi-model functionality, the application uses the following technique:

1. **Sentiment Analysis**: Utilizes a custom fine-tuned model for sentiment analysis. The model was fine-tuned using a [`public dataset`](https://www.kaggle.com/datasets/nursyahrina/chat-sentiment-dataset) of sentiment analysis and a pre-trained model `gemini-1.5-flash-001-tuning`.
2. **Text Generation**: Utilizes Google Generative AI's `gemini-1.5-flash` model for text generation.
3. **Text History**: Utilizes the Google Generative AI's `start_chat()` method to start a conversation with the model while storing the chat history.
4. **Image Generation**: Utilizes Pollinations AI's `flux` model for image generation.
5. **Image Understanding**: Utilizes Google Generative AI's `gemini-1.5-flash` model existing features for image understanding.
6. **Image Variation**: Utilizes Google Generative AI's `gemini-1.5-flash` model to understand when the user requests image variation, and then prompts Pollinations AI's `flux` model for image variation.
7. **Contextual Integrity**: Utilizes multiple Google Generative AI's `gemini-1.5-flash` models to always keep a complete contextual understanding of the entire conversation at all times for both text and image. Also uses a extra model to keep understanding of the contextual sentiment of the overall conversation. This helps to increase the accuracy of the sentiment analysis.

## Prerequisites

### Libraries and Tools

1. Python 3.12
2. Required Libraries:
   - `os`
   - `pillow`
   - `streamlit`
   - `pollinations`
   - `random`
   - `datetime`
   - `dotenv`
   - `google.generativeai`
   - `seaborn`
3. Environment Variables:
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
   SEMTIMENTIAL_MODEL=<Full name of the fine-tuned sentiment model>
   ```

### Fine-Tuning Sentiment Model

1. Confirm the dataset `chat_dataset.csv` is present in the `sentimental_chatbot/dataset/` directory. This dataset contains a CSV file containing messages and their corresponding sentiments.
2. Run the `tuned_model.py` script to fine-tune a model for sentiment analysis.
3. After the model is completed training, update the environment with the fine-tuned model's name. [The model name will be printed in the console]
4. The fine-tuned model is will be this format `tunedModels/{fine-tuned model name}`
   - For example: `tunedModels/sentimental-bot-xxxxxxxxxxxx` xxxxxxxxxxxx will be different for every model.
5. Set the `SEMTIMENTIAL_MODEL` environment variable to the fine-tuned model name. with the `tunedModels/sentimental-bot-xxxxxxxxxxxx`

### Usage

1. Run the Streamlit application:

    ```bash
    streamlit run sentimental_chatbot\src\app.py
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

- **`app.py`**: Main Streamlit application file used to run the web application.
- **`tuned_model.py`**: Script to fine-tune a model for sentiment analysis.
- **`chat_dataset.csv`**: Dataset used for training the sentiment analysis model. This dataset contains messages and their corresponding sentiments.

---

## Implementation

- **Custom CSS Styling**:
  - Provides a customized user interface with modified layouts for chat and file upload components.
  - Modifies the layout of Streamlit components for better usability.
  - Hides unnecessary elements like default file uploader labels.

- **Model Initialization**:
  - Configures Google Generative AI models for both text and image generation.
  - Pollinations AI is used for generating images with specific parameters (e.g., resolution, seed).
  - Initializes sentiment, text and image generation models:
    - **Sentiment Model**: fine-tuned model of `gemini-1.5-flash-001-tuning`
    - **Text Model**: `gemini-1.5-flash`
    - **Image Model**: `flux` (Pollinations AI).
  - Initializes text, image, and context models in the Streamlit session state.

- **Fine-Tuning Sentiment Model**:
  - Loads a dataset [(`chat_dataset.csv`)](https://www.kaggle.com/datasets/nursyahrina/chat-sentiment-dataset) containing messages and their corresponding sentiments and formats data into input-output pairs for training.
  - Fine-tunes a base model `gemini-1.5-flash-001-tuning` by Google Generative AI APIs with specified hyperparameters (20 epochs, batch size of 5, learning rate of 0.001).
  - Visualizes the training progress with a loss function curve using Seaborn.
  - Deploys the fine-tuned model and verifies it with sample outputs.

- **Sentiment Analysis into Chat**:
  - Utilizes the fine-tuned sentiment model to classify user input and apply color coding for sentiment display.
  - The fine-tuned model generates an output in [positive, neutral, negative] format. Each sentiment is assigned a color when displayed. Positive: *Green*, Neutral: *Blue*, Negative: *Red* to better represent the sentiment.
  - Uses multi-model verification of the sentiment generated to ensure its validity.
  - Also understands images and can classify them into sentiments using a multi-model pipeline.

- **Text Generation and Chat Functionality**:
  - Stores chat history in the Streamlit session state.
  - Maintains a contextual understanding of the conversation for accurate responses.
  - Displays previous messages with roles (user/AI) and sentiment classification.

- **Image Upload and Generation**:
  - Allows users to upload images or request image generation.
  - Generates images based on user prompts or variations of previously generated images.
  - Displays uploaded and generated images in the chat interface.
  - Saves uploaded and generated images locally with timestamped filenames for easy identification.

- **Prompt Processing and Contextual Understanding**:
  - Detects user input requests for image generation or variations of existing images.
  - Prompts the appropriate model for text or image generation based on the context of the conversation.

- **Error Handling and Logging**:
  - Handles API errors and other exceptions gracefully, displaying error messages to the user.
  - Logs key events like model initialization, image saving, and prompt processing to the console for debugging purposes.

---

## Conclusion

The Multi-Modal Sentimental Chatbot combines advanced AI capabilities to deliver an engaging and interactive user experience with sentiment analysis.
