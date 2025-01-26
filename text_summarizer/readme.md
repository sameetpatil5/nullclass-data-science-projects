# Text Summarization Project

This document outlines the documentation, setup and explanation of the code used for development of the Text Summarization project. The project is implemented in **Python** using **Natural Language Toolkit (NLTK)**, and basic Natural Language Processing (NLP) techniques. An Web application is created using the **Streamlit** framework to provide a user-friendly interface for summarizing text.

---

## Overview

Text summarization is the process of condensing a large body of text into a shorter version while retaining the most important information. This project implements two approaches to text summarization:

1. **Frequency-based Summarization**: Using word frequencies to identify and rank sentences for inclusion in the summary.
2. **TF-IDF-based Summarization**: Using Term Frequency-Inverse Document Frequency (TF-IDF) scores to rank sentences for summarization.

Both the approaches have been implemented in the  **`text_summarizer_training.ipynb`** file. The notebook contains the functions required to build the text summarization function. It summarizes the text defined in `text.txt` file to `summary.txt` and `tf-idf_summary.txt` using Frequency-based and TF-IDF-based approaches respectively.

The UI can be found in `text_summarizer/src` is developed in Streamlit and is available in the **`app.py`**. The user can interact with the UI to input text and the summarized text output will be displayed. The app uses **`text_summarizer.py`**, which contains the same functions from the **`text_summarizer_training.ipynb`** but in a python file.

---

## Prerequisites

### Libraries and Tools

- Python 3.12
- Required Python libraries:
  - `nltk`
  - `spacy`
  - `sklearn`
  - `streamlit`
- Download the required NLTK data using `nltk.download("punkt_tab")` and `nltk.download("stopwords")`.
- Download the required SpaCy model using `python -m spacy download en_core_web_sm`.

### Installation

1. Setup a virtual environment

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install the required libraries:

   ```bash
   pip install nltk spacy scikit-learn streamlit
   ```

3. Download the necessary NLTK resources and SpaCy English language model:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

   ```bash
   python -m spacy download en_core_web_sm
   ```

### Usage

1. Run the Streamlit app:

   ```bash
   streamlit run text_summarizer\src\app.py
   ```

---

## Files

- **`text.txt`**: Input file containing the text to be summarized.
- **`summary.txt`**: Output file for the frequency-based summary.
- **`tf-idf_summary.txt`**: Output file for the TF-IDF-based summary.
- **`summarizer.ipynb`**: Jupyter Notebook containing the implementation.
- **`text_summarizer.py`**: Module containing the text summarization functions.
- **`app.py`**: Streamlit app.

---

## Implementation

### 1. Frequency-Based Summarization

1. **Create Frequency Table**:
   - Tokenize the input text into words.
   - Remove stopwords and apply stemming using Porter Stemmer.
   - Count the frequency of each word.

2. **Score Sentences**:
   - Tokenize the text into sentences.
   - Calculate a score for each sentence based on the word frequencies.

3. **Determine Threshold**:
   - Compute the average sentence score.
   - Set a threshold to select sentences for the summary.

4. **Generate Summary**:
   - Include sentences with scores above the threshold.

#### Frequency-Based Summarization Functions

1. **`create_frequency_table(text_string) -> dict`**:
   - Creates a frequency table of stemmed words, excluding stopwords.

2. **`score_sentences(sentences, freq_table) -> dict`**:
   - Scores sentences based on word frequencies.

3. **`find_average_score(sentence_value) -> int`**:
   - Computes the average score of all sentences.

4. **`generate_summary(sentences, sentence_value, threshold) -> str`**:
   - Generates the summary based on a threshold score.

5. **`summarize_text(text) -> str`**:
   - Orchestrates the summarization process

### 2. TF-IDF-Based Summarization

1. **Tokenize Sentences**:
   - Use SpaCy to split the text into sentences.

2. **Calculate TF-IDF Scores**:
   - Use `TfidfVectorizer` to compute the TF-IDF scores for each sentence.

3. **Rank Sentences**:
   - Sort sentences by their TF-IDF scores in descending order.

4. **Generate Summary**:
   - Select the top `n` sentences for the summary.

#### TF-IDF-Based Summarization Function

1. **`summarize_with_tfidf(text, top_n=5)`**:
   - Summarizes the text using TF-IDF scores and selects the top `n` sentences.

---

## Conclusion

The Text Summarization is a classic implementation of NLP using frequency-based and TF-IDF-based approaches. The project provides a user-friendly interface for summarizing text, making it an effective tool for text summarization.
