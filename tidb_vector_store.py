from dotenv import load_dotenv
load_dotenv()

import os
import threading

from langchain_community.vectorstores import TiDBVectorStore
from langchain_openai import OpenAIEmbeddings

from utils import load_tickers

_cluster1_tickers = load_tickers('cluster1_tickers.txt')
_cluster2_tickers = load_tickers('cluster2_tickers.txt')
all_tickers = sorted(_cluster1_tickers + _cluster2_tickers)

_embeddings = OpenAIEmbeddings(
                api_key=os.getenv('OPENAI_EMBEDDING_API_KEY'),
                model=os.getenv('OPENAI_EMBEDDING_MODEL'),
                base_url=os.getenv('OPENAI_EMBEDDING_BASE_URL'),
                dimensions=int(os.getenv('OPENAI_EMBEDDING_MODEL_DIMS')))

def _get_vector_store(url_env_var: str) -> TiDBVectorStore:
    print(f'Initializing vector store {url_env_var}...')
    vectorstore = TiDBVectorStore.from_existing_vector_table(
                    embedding=_embeddings,
                    connection_string=os.getenv(url_env_var),
                    table_name=os.getenv('TIDB_TABLE_NAME'))
    return vectorstore

_cached_vector_store1 = _get_vector_store('TIDB_DATABASE_URL1')
_cached_vector_store2 = _get_vector_store('TIDB_DATABASE_URL2')

_lock1 = threading.Lock()
_lock2 = threading.Lock()

def get_cached_vector_store(ticker: str) -> TiDBVectorStore:
    global _cached_vector_store1
    global _cached_vector_store2

    if ticker in _cluster1_tickers:
        wanted_vector_store = _cached_vector_store1
        url_env_var = 'TIDB_DATABASE_URL1'
        lock = _lock1
    elif ticker in _cluster2_tickers:
        wanted_vector_store = _cached_vector_store2
        url_env_var = 'TIDB_DATABASE_URL2'
        lock = _lock2
    else:
        assert False, f'Ticker {ticker} not associated with a cluster.'

    with lock:
        try: # ping the vector store to check if the connection is still alive
            _result = wanted_vector_store.similarity_search('', k=0)
            pass
        except Exception as e:
            print(f'Error: {e}')
            wanted_vector_store = _get_vector_store(url_env_var)
            if url_env_var == 'TIDB_DATABASE_URL1':
                _cached_vector_store1 = wanted_vector_store
            else:
                assert url_env_var == 'TIDB_DATABASE_URL2'
                _cached_vector_store2 = wanted_vector_store
    return wanted_vector_store
