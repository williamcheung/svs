from dotenv import load_dotenv

from tidb_vector_store import get_cached_vector_store
load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from llm import get_llm_sambanova, get_llm_openai

def ask_question(ticker: str, question: str, filter={}) -> str:
    if ticker:
        filter['ticker'] = ticker
    search_kwargs = {'filter': filter, 'k': 20}
    retriever = get_cached_vector_store().as_retriever(search_kwargs=search_kwargs)

    # define the RAG prompt
    template = '''Answer the question based only on the following context:
    {context}
    Question: {question}

    NOTE: If the answer is not found in the context or cannot be inferred from it, say "I can't find this in the financial statements."

    Your answer is for a non-technical investor so do not mention the AI/ML jargon word "context"; instead refer to "financial statements" if needed.

    Begin your answer with "Here's what I found for <FULL COMPANY NAME> based on their latest financial statements:" and then go directly into your answer.

    Also give the links (with titles) to the financial statements used at the end.

    Make any headings bold and numbered, for example: **1. Revenue**
    '''
    prompt = ChatPromptTemplate.from_template(template)

    # define the RAG model
    model = get_llm_sambanova()
    chain = _create_rag_chain(retriever, prompt, model)

    print(f'LANGCHAIN RAG Q: [{ticker}] {question}')
    try:
        answer = chain.invoke(question)
    except Exception as e: # handle rate limit errors, etc. by trying the other model
        print(f'Error: {e}')
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