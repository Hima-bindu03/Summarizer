# Blog Summarizer with LangChain Groq & Streamlit

A Streamlit app that summarizes any web article using LangChain Groq API.  
It supports two summarization methods: **Stuff** and **Map-Reduce**, allowing easy comparison of both techniques.

## Features

- Input any blog or article URL.
- Get summaries via Stuff method or Map-Reduce method.
- Clean and interactive UI built with Streamlit.

## Installation

git clone https://github.com/<your-username>/summarizer-app.git

cd summarizer-app

pip install -r requirements.txt

### Usage
Create a .env file in the project root with:
GROQ_API_KEY=your_groq_api_key_here

### Run the Streamlit App:
streamlit run app.py


