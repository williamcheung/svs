from dotenv import load_dotenv
load_dotenv()

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from llm import get_llm_sambanova, get_llm_openai
from tidb_vector_store import get_cached_vector_store
from utils import load_prompt

MAX_VECTORS_RETURNED = int(os.getenv('MAX_VECTORS_RETURNED'))

def ask_question(ticker: str, question: str, filter={}) -> str:
    if ticker:
        filter['ticker'] = ticker_in_data_file(ticker)
    search_kwargs = {'filter': filter, 'k': MAX_VECTORS_RETURNED}
    retriever = get_cached_vector_store(ticker).as_retriever(search_kwargs=search_kwargs)

    # define the RAG prompt
    template = load_prompt('rag_prompt.txt')
    prompt = ChatPromptTemplate.from_template(template)

    # define the RAG model
    model = get_llm_sambanova()
    chain = _create_rag_chain(retriever, prompt, model)

    print(f'LANGCHAIN RAG Q: [{ticker}] {question}')
    try:
        answer = chain.invoke(question)
    except Exception as e: # handle rate limit errors, etc. by trying the other model
        print(f'Error for [{ticker}] {question}: {e}')
        print('Trying other model ...')
        model = get_llm_openai()
        chain = _create_rag_chain(retriever, prompt, model)
        try:
            answer = chain.invoke(question)
        except Exception as e:
            print(f'Error: {e}')
            answer = "I can't find this in the financial statements."

    print(f'A: {answer}')
    return answer

def _create_rag_chain(retriever, prompt, model):
    chain = (
        RunnableParallel({'context': retriever, 'question': RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )
    chain = chain.with_types(input_type=str)
    return chain

def ticker_in_data_file(ticker: str) -> str:
    return ticker.replace('.', '-') # data file uses "-" instead of "." in actual symbol

if __name__ == '__main__':
    # test usage
    tickers = ['WBD']
    tickers = ['PARA']
    questions = [
        'How were (1) the Revenue, (2) the Net income, (3) Earnings Per Share? Give as much detail as possible for each. Also give the links (with title) to the financial statements used at the end.',
    ]
    filter = {'date': {'$gt': '2024-09-30'}, 'form_type': {'$in': ['8-K', '10-Q']}}

    for ticker in tickers:
        for question in questions:
            answer = ask_question(ticker, question, filter)
