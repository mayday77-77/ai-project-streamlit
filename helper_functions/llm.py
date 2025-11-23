import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken


def openai_client():
    if load_dotenv('.env'):
        # for local development
        openai_key = os.getenv('OPENAI_API_KEY')
    else:
        # for streamlit cloud
        openai_key = st.secrets['OPENAI_API_KEY']

    return OpenAI(api_key=openai_key)


# This function is for calculating the tokens given the "message"
# ⚠️ This is simplified implementation that is good enough for a rough estimation
def count_tokens(text):
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    return len(encoding.encode(text))


def count_tokens_from_message(messages):
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    value = ' '.join([x.get('content') for x in messages])
    return len(encoding.encode(value))
