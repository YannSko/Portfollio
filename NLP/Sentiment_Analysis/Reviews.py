import streamlit as st
from transformers import pipeline

def run_sentiment_analysis(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)
    return result[0]['label']


def app():
    st.title('Sentiment Analysis App')
    text_input = st.text_input('Enter some text:')
    if st.button('Analyze'):
        result = run_sentiment_analysis(text_input)
        st.write(f'The sentiment of the text is {result}')

if __name__ == '__main__':
    app()

