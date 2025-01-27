# NullClass Internship Projects

This document is prepared to provide an overview of the projects _**I**_ developed as part of my internship at **NullClass**. These projects reflect my understanding and implementation of advanced AI capabilities, adhering to the guidelines and conditions set forth by NullClass.

---

## Overview

This repository contains the submission of tasks assigned during my internship at **NullClass**. THere are three distinct AI-driven projects, which were exclusively developed for **NullClass**. These projects, their methodologies, and implementations are the intellectual property of NullClass. Any unauthorized sharing or usage is strictly prohibited.

### Projects

1. **Text Summarization**  _(Task 1)_  
   A tool for extractive summarization using frequency-based and TF-IDF-based approaches.
2. **Multi-Modal Chatbot**  _(Task 2)_  
   A chatbot capable of text and image generation with contextual understanding.
3. **Sentimental Multi-Modal Chatbot**  _(Task 3)_  
   An advanced version of the multi-modal chatbot integrated with sentiment analysis for enhanced interaction.

---

## Project Summaries

### 1. **Text Summarization**

- **Objective**: Implement an extractive summarization technique to condense lengthy texts while retaining essential information.
- **Techniques Used**:
  - Frequency-Based Summarization
  - TF-IDF-Based Summarization
- **Key Features**:
  - Summarizes text input via a user-friendly Streamlit UI.
  - Outputs summaries to `summary.txt` (frequency-based) and `tf-idf_summary.txt` (TF-IDF-based).
- **Implementation**:
  - Tokenization, stemming, and stopword removal for frequency-based scoring.
  - TF-IDF vectorization for sentence ranking.
- **Files**:
  - `text_summarizer_training.ipynb`, `text_summarizer.py`, `app.py`
- **Requirements**:
  - Python 3.12
  - Libraries: `nltk`, `spacy`, `scikit-learn`, `streamlit`

---

### 2. **Multi-Modal Chatbot**

- **Objective**: Develop a chatbot with text and image generation capabilities using Google Generative AI and Pollinations AI.
- **Key Features**:
  - Generates images and text responses using Google Generative AI and Pollinations AI based on user input.
  - Maintains contextual understanding across conversations.
  - Image understanding and variation capabilities.
- **Implementation**:
  - Text generation using `gemini-1.5-flash`.
  - Image generation via Pollinations AI's `flux` model.
  - Custom CSS styling for an intuitive Streamlit UI.
- **Files**:
  - `app.py`
- **Requirements**:
  - Python 3.12
  - Libraries: `streamlit`, `pollinations`, `google-generativeai`
  - **Environment Variable**: `GEMINI_API_KEY`

---

### 3. **Sentimental Multi-Modal Chatbot**

- **Objective**: Enhance the multi-modal chatbot with sentiment analysis for better user interaction.
- **Key Features**:
  - Sentiment analysis using a fine-tuned `gemini-1.5-flash-001-tuning` model.
  - Displays sentiment with color-coded outputs (Positive: Green, Neutral: Blue, Negative: Red).
  - Handles both text and image inputs with sentiment classification.
- **Implementation**:
  - Fine-tuned sentiment model using a public dataset.
  - Sentiment-aware responses and contextual understanding.
  - Image and text generation with integrated sentiment analysis.
- **Files**:
  - `app.py`, `tuned_model.py`, `chat_dataset.csv`
- **Requirements**:
  - Python 3.12
  - Libraries: `streamlit`, `pollinations`, `google-generativeai`
  - **Environment Variables**: `GEMINI_API_KEY`, `SEMTIMENTIAL_MODEL`

---

## Common Requirements

- **Python Version**: 3.12
- **Setup**:

    1. Create a virtual environment:

       ```bash
       python -m venv .venv
       .venv\Scripts\activate
       ```

    2. Install dependencies using `pip install -r requirements.txt`.
    3. Set up environment variables in a `.env` file.

---

## Important Notes

1. This project requires a **continuous internet connection** for API interactions.
2. **Gemini API Key**: Users must provide their own `GEMINI_API_KEY`.
3. The fine-tuned models are hosted on Google servers. Hence, It is not possible share the fine-tuned model file or link. It is required to train these models independently as they cannot be shared directly.
4. Any changes to the APIs (e.g., Gemini or Pollinations) may affect the project’s functionality. I am not responsible for such changes impacting performance.
5. All NullClass guidelines and conditions have been strictly adhered to.

---

## NullClass Conditions Summary

1. Tasks are evaluated on both **accuracy** and **performance** (minimum 70% accuracy required).
2. Projects must use **custom-trained models**; pre-trained models alone are disallowed.
3. Submissions must include:
   - `requirements.txt`
   - Model training files (`.ipynb`)
   - Saved models (shared via Google Drive if large)
   - Source code
   - Organized folder structure
4. Plagiarism or sharing work online will result in disqualification.
5. Timely daily reports and final submission within the given timeframe (02-01-2025 to 02-02-2025) are mandatory.

---

## Acknowledgment

This repository reflects my efforts to meet NullClass's high standards for AI development. It demonstrates proficiency in NLP, sentiment analysis, and multi-modal AI technologies, adhering to the professional and technical guidelines set forth by NullClass.
