from dotenv import load_dotenv
load_dotenv()

import os

from langchain_community.vectorstores import TiDBVectorStore
from langchain_openai import OpenAIEmbeddings
import threading

embeddings = OpenAIEmbeddings(
                api_key=os.getenv('OPENAI_EMBEDDING_API_KEY'),
                model=os.getenv('OPENAI_EMBEDDING_MODEL'),
                base_url=os.getenv('OPENAI_EMBEDDING_BASE_URL'),
                dimensions=int(os.getenv('OPENAI_EMBEDDING_MODEL_DIMS')))

def _get_vector_store() -> TiDBVectorStore:
    vectorstore = TiDBVectorStore.from_existing_vector_table(
                    embedding=embeddings,
                    connection_string=os.getenv('TIDB_DATABASE_URL'),
                    table_name=os.getenv('TIDB_TABLE_NAME'))
    return vectorstore

cached_vector_store = _get_vector_store()
lock = threading.Lock()
def get_cached_vector_store() -> TiDBVectorStore:
    with lock:
        global cached_vector_store
        try: # ping the vector store to check if the connection is still alive
            result = cached_vector_store.similarity_search('', k=0)
            pass
        except Exception as e:
            print(f'Error: {e}')
            print('Reinitializing vector store...')
            cached_vector_store = _get_vector_store()
        return cached_vector_store
