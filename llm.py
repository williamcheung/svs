from dotenv import load_dotenv
load_dotenv()

import os

from langchain_openai import ChatOpenAI

def get_llm_sambanova():
    return ChatOpenAI(temperature=0, api_key=os.getenv('OPENAI_API_KEY2'),
                      model=os.getenv('OPENAI_MODEL2'), base_url=os.getenv('OPENAI_BASE_URL2'),
                      max_tokens=int(os.getenv('MAX_TOKENS2')))

def get_llm_openai():
    return ChatOpenAI(temperature=0, api_key=os.getenv('OPENAI_API_KEY'),
                      model=os.getenv('OPENAI_MODEL'), base_url=os.getenv('OPENAI_BASE_URL'),
                      max_tokens=int(os.getenv('MAX_TOKENS')))
