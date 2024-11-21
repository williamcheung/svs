from dotenv import load_dotenv
load_dotenv()

import os

from langchain_community.vectorstores import TiDBVectorStore
from langchain_openai import OpenAIEmbeddings
import threading

_cluster1_tickers = ['AAPL','MSFT','NVDA','AMZN','GOOG','GOOGL','META','BRK.B','LLY','TSLA','AVGO','WMT','JPM','UNH','V','XOM','MA','PG','JNJ','COST','ORCL','HD','ABBV','KO','BAC','MRK','NFLX','CVX','ADBE','PEP','TMO','CRM','TMUS','AMD','LIN','ACN','MCD','ABT','PM','DHR','CSCO','IBM','WFC','TXN','VZ','GE','QCOM','AXP','NOW','INTU','AMGN','ISRG','NEE','PFE','GS','CAT','SPGI','RTX','DIS','MS','T','CMCSA','UNP','PGR','UBER','AMAT','LOW','SYK','LMT','TJX','HON','BLK','BKNG','ELV','REGN','COP','BSX','VRTX','PLD','NKE','CB','MDT','SCHW','ETN','C','MMC','ADP','PANW','AMT','UPS','ADI','BX','DE','KKR','SBUX','ANET','MDLZ','BA','CI','HCA','FI','GILD','BMY','SO','MU','KLAC','LRCX','ICE','MO','SHW','DUK','MCO','CL','ZTS','WM','GD','INTC','CTAS','EQIX','CME','TT','WELL','NOC','AON','PH','CMG','ABNB','ITW','MSI','APH','TDG','PNC','SNPS','CVS','ECL','PYPL','USB','MMM','FDX','TGT','CDNS','BDX','EOG','MCK','AJG','CSX','ORLY','RSG','MAR','CARR','PSA','AFL','DHI','APD','CRWD','ROP','NXPI','NEM','NSC','FCX','FTNT','SLB','TFC','EMR','GEV','AEP','ADSK','TRV','O','CEG','MPC','COF','WMB','OKE','PSX','AZO','GM','HLT','MET','SPG','SRE','CCI','KDP','ROST','BK','PCAR','MNST','KMB','LEN','ALL','DLR','OXY','D','PAYX','CPRT','GWW','AIG','KMI','CHTR','COR','URI','JCI','STZ','FIS','KVUE','TEL','MSCI','IQV','KHC','FICO','LHX','RCL','VLO','AMP','F','PCG','ACGL','GIS','HUM','NDAQ','PRU','HSY','MPWR','CMI','ODFL','MCHP','PEG','A','EW','HES','IDXX','FAST','VRSK','GEHC','EXC','CTVA','SYY','HWM','EA','AME','IT','CTSH','KR','YUM','CNC','EXR','PWR','EFX','OTIS','RMD','ED','DOW','VICI','XEL','IR','GRMN','GLW','CBRE','HIG','DFS','BKR','NUE','EIX','DD','HPQ','AVB','CSGP','IRM','FANG','TRGP','XYL','EL','MLM','LYB','VMC','LULU','WEC','WTW','ON','BRO','LVS','MRNA','PPG','TSCO','ROK','MTD','EBAY','BIIB','CDW','WAB','EQR','AWK','ADM','MTB','NVR','FITB','DAL','GPN','DXCM','K','AXON','CAH','TTWO','PHM','ANSS','VLTO','VTR','IFF','ETR','DVN','CHD','DTE','SBAC','VST','FE','FTV','HAL','KEYS','TYL','STT','DOV','BR','ES','STE','RJF','ROL','SMCI','PPL','NTAP','TSN','SW','TROW','HPE','DECK','WRB','AEE','MKC','CBOE','WY','FSLR','WST','BF.B','INVH','LYV','GDDY','COO','WDC','CINF','ZBH','CPAY','STX','HBAN','BBY','ATO','ARE','LDOS','CMS','RF','CLX','CCL','HUBB','TER','PTC','BAX','TDY','WAT','BALL','BLDR','OMC','ESS']
_cluster2_tickers = ['HOLX','LH','SYF','GPC','MOH','EQT','CFG','MAA','DRI','FOXA','APTV','PFG','PKG','ULTA','J','WBD','CNP','LUV','DG','HRL','VRSN','FOX','NTRS','AVY','L','JBHT','EXPE','EXPD','DGX','STLD','ZBRA','MAS','CTRA','EG','IP','ALGN','FDS','TXT','NRG','AMCR','UAL','SWKS','GEN','CAG','KIM','DOC','CPB','NWS','PODD','LNT','NWSA','UHS','KEY','NI','IEX','MRO','SWK','DPZ','UDR','RVTY','SNA','DLTR','AKAM','PNR','CF','NDSN','BG','ENPH','EVRG','REG','VTRS','TRMB','POOL','CE','CPT','SJM','JNPR','DVA','KMX','JKHY','INCY','CHRW','HST','EPAM','BXP','ALLE','IPG','FFIV','JBL','TAP','SOLV','TFX','AES','EMN','TECH','AOS','CTLT','RL','MGM','LKQ','HII','BEN','PNW','AIZ','QRVO','FRT','MKTX','CRL','TPR','HAS','MHK','MTCH','GL','APA','ALB','PAYC','LW','BIO','DAY','HSIC','GNRC','WYNN','MOS','CZR','NCLH','WBA','FMC','BWA','AAL','IVZ','PARA','BBWI','ETSY']
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
