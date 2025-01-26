# Internship Report for NullClass

## Intern Details

- Name: **Sameet Patil**
- Email: <sameetpatil5@gmail.com>
- Internship Duration: _2^nd^ January, 2025_ to _2^nd^ February, 2025_
- Submission Date: _26^th^ January, 2025_
- Training Program Offer Letter: [NullClass Data Science Training Offer Letter](./assets/sameetpatil_nullclass_datascience_training_ol_signed.pdf)
- Training Program Certification: [NullClass Data Science Training Certificate](./assets/sameetpatil_nullclass_datascience_training_certificate.pdf)
- Internship Offer Letter: [NullClass Internship Offer Letter](./assets/sameetpatil_nullclass_datascience_intern_ol_signed.pdf)

---

## Introduction

<p align="justify"> This report is prepared to documents my internship experience at **NullClass**, highlighting the tasks undertaken, skills developed, and outcomes achieved during the internship. The projects assigned allowed me to explore advanced AI concepts, including natural language processing, sentiment analysis, and multi-modal AI integration, aligning with my learning objectives and career aspirations.

---

## Background

<p align="justify"> The internship program by NullClass provided a platform for me to gain hands-on experience in AI-driven projects. This internship focused on implementing AI solutions for various tasks assigned by NullClass. The internship's primary objective was to enhance my understanding of AI models and their applications, while contributing to meaningful projects.

### Projects

1. **Task 1: _Text Summarization_**: <p align="justify"> Implement an extractive summarization technique using NLP to condense lengthy texts while retaining essential information.
2. **Task 2: _Multi-Modal Chatbot_**: <p align="justify"> A chatbot capable of text and image generation with contextual understanding.
3. **Task 3: _Sentimental Multi-Modal Chatbot_**: <p align="justify"> An advanced version of the multi-modal chatbot integrated with sentiment analysis for enhanced interaction.

### Role

<p align="justify"> The internship required proficiency in machine learning, natural language processing, Large Language Models (LLMs), sentiment analysis, image generation, and API integration. My role was to independently research, develop, and evaluate solutions while ensuring the models met the stipulated accuracy and performance metrics.

---

## Learning Objectives

1. <p align="justify"> Understand the fundamentals of NLP and how an Large Language Model interprets a language.
2. <p align="justify"> Gain practical experience in developing AI applications using Python and relevant frameworks.
3. <p align="justify"> Enhance problem-solving skills by addressing challenges in AI model training and integration.
4. <p align="justify"> Learn to document and present technical projects professionally.
5. <p align="justify"> Implement projects based on NLP and AI model especially LLMs to achieve real-world applications.
6. <p align="justify"> Develop a complex chatbot application and also fine-tunning LLMs for text classification.

---

## Activities and Tasks

### 1. Text Summarization

- Implemented NLP based text summarization using:
  - Frequency-based techniques.
  - TF-IDF-based techniques.
- Developed a Streamlit-based user interface for summarization.
- Key Deliverables:
  - Summarized text outputs (`summary.txt`, `tf-idf_summary.txt`) from the text input in `text.txt`.
  - Python scripts: `text_summarizer_training.ipynb`, `text_summarizer.py`, and `app.py`.

### 2. Multi-Modal Chatbot

- Created a chatbot integrating text and image generation capabilities.
- Utilized Google Generative AI and Pollinations AI for:
  - Text responses and conversational chat.
  - Image generation and variation.
- Key Deliverables:
  - Streamlit-based chatbot interface (`app.py`).

### 3. Sentimental Multi-Modal Chatbot

- Enhanced the chatbot with sentiment analysis capabilities.
- Fine-tuned a sentiment model using a [`public dataset`](https://www.kaggle.com/datasets/nursyahrina/chat-sentiment-dataset) and Google Generative AI APIs Fine-tuning.
- Integrated sentiment classification into chatbot interactions.
- Key Deliverables:
  - Fine-tunning model python scripts: (`tuned_model.py`).
  - Streamlit-based Sentiment-aware chatbot interface (`app.py`).

---

## Skills and Competencies

During the internship, I gained and demonstrated several skills and competencies:

1. **Technical Skills**:
   - Proficiency in Python programming.
   - Hands-on experience with libraries like `streamlit`, `nltk`, `spacy`, and `scikit-learn`.
   - Knowledge of AI APIs (Google Generative AI, Pollinations AI) and their Python-SDKs (`google-generativeai`, `pollinations`).
   - Fine-tuning AI models for specific tasks using Google Generative AI.
2. **Soft Skills**:
   - Problem-solving and critical thinking.
   - Effective documentation and reporting with time management.
   - Successfully organized tasks and managed time to meet strict deadlines.
3. **Project Specific Skills**:
   - Understanding NLP and Large Language Models.
   - Proper implementation of Generative AI APIs and Python libraries for the same.
   - Independently solving problems and developing projects with a focus on high scope and accuracy.

---

## Feedback and Evidence

### Feedback

- <p align="justify"> As mentor support was not available during the internship, I could not gather any specific feedback from NullClass regarding the projects.
- <p align="justify"> I have submitted all deliverables, including source code, trained models, and documentation.
- <p align="justify"> As self-reflection, I found the integration of multi-modal capabilities challenging initially. However, with research and iterative testing, I was able to achieve seamless text and image integration.
- <p align="justify"> While testing it was revealed that the sentiment analysis chatbot could sometimes misinterpret sarcasm and other language nuances which still remains an open issue.

>_The accuracy is still high_

### Evidence

- Demonstrated working applications in this [YouTube video]() and [GitHub repository](https://github.com/sameetpatil5/nullclass-data-science-projects.git).
- Made proper documentation of the all the projects.
- Submitted all deliverables, including source code, trained models, and documentation.
- Screenshots of the projects:

---

## Challenges and Solutions

### Challenges

1. **NLP Techniques**:
   - <p align="justify"> Implementing a relatively accurate NLP techniques for text summarization was difficult to identify. Each technique had its own characteristics and approaches.

2. **Multi-Model Contextual Understanding**:
   - <p align="justify"> Making the model understand the context of the conversation at all times was a challenge. The image model had to be given context of the conversation to understand the user's intent. The text model had to be given context of the image model to understand the user's intent.
   - <p align="justify"> Because of different text and image models used, the chatbot had to be able to handle both text and image inputs while maintaining the context of the conversation.

3. **Sentiment Analysis Accuracy**:
   - <p align="justify"> Ensuring high accuracy in sentiment classification across varied inputs was a challenge. The sentiment model had to be fine-tuned with a public dataset to improve accuracy. But the dataset was very general and could not classify all the inputs accurately.

4. **Streamlit state management**:
   - <p align="justify"> <p align="justify"> Managing state in Streamlit was a challenge. The chatbot had to be able to store and retrieve information from the Streamlit session state. The chat history and image history was needed to be maintained properly as it may imapct the following prompts and responses.

### Solutions

1. **NLP Techniques**:
   - <p align="justify"> Implemented various NLP techniques for text summarization, including frequency-based and TF-IDF-based approaches. And manually verified the accuracy of the outputs. This made sure that the outputs were relevant and accurate.

2. **Multi-Model Contextual Understanding**:
   - <p align="justify"> Utilized multiple text and image models to provide context to the chatbot. This allowed the chatbot to understand the user's intent and respond accordingly.
   - <p align="justify"> Implemented these models in a way that the chatbot able to prompt itself and other models to respond accurately and also maintain the context of the conversation.

3. **Sentiment Analysis Accuracy**:
   - <p align="justify"> This was solved in a similar way that the multi-model contextual understanding was solved. The chatbot was referred multiple models for the sentiment analysis, one was the fine-tuned model, and the others included models to verify its accuracy, analyze the sentiment of the image and the overall sentiment of the conversation using the context of the conversation.

4. **Streamlit state management**:
   - <p align="justify"> This challenge was solved by watching tutorials and reading documentation to learn how to store and retrieve information in Streamlit.
   - <p align="justify"> This improved my understanding of Streamlit and its capabilities which in turn allowed me to implement the chatbot with the desired functionality.

---

## Outcomes and Impact

1. **Technical Outcomes**:
   - Developed three fully functional AI-driven applications.
   - Enhanced understanding of multi-modal AI and sentiment analysis.
   - Gained hands-on experience with Python and relevant frameworks to build UIs for AI-driven applications quickly.
   - Learnt to fine-tune AI models for specific tasks.
2. **Personal Growth**:
   - Improved problem-solving and technical documentation skills.
   - Gained confidence in handling complex AI projects independently.
   - Learnt to manage time effectively to meet deadlines.
   - Learnt to self-reflect and learn from my mistakes.
3. **Professional Growth**:
   - Improved my soft skills and project-specific skills.
   - Learnt to document and present technical projects professionally.
   - Gained valuable experience in working with NullClass and their assigned projects.

---

## Conclusion

<p align="justify"> Completing the tasks assigned during the internship at NullClass has been a valuable learning experience. The tasks helped me develop a deeper understanding of latest AI technologies and their applications, equipping me with skills crucial for my future endeavors. I am grateful for the opportunity to contribute to innovative projects and look forward to applying these learnings in my career.
