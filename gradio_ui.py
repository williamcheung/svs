import asyncio
import json
import gradio as gr

from concurrent.futures import ThreadPoolExecutor
from gradio import ChatMessage
from langchain_core.prompts import PromptTemplate
from typing import Generator

from langchain_tidb_rag import ask_question
from llm import get_llm_sambanova
from utils import load_prompt

TITLE = 'Stock vs. Stock'
COMP_A = 'Company A'
COMP_B = 'Company B'
INVESTMENT_FACTOR = 'investment factor'

GREETING = f'''
Welcome to <b>{TITLE}</b>. Choose 2 <u>S&P 500 stocks</u> to compare and one of <u>10 {INVESTMENT_FACTOR}s</u>.
When ready, click <u>Compare</u>.

Caveat: <b>{TITLE}</b> recommendations are for <i>educational purposes only</i>. Getting rich💰not guaranteed.
'''

tickers = ['AAPL','MSFT','NVDA','AMZN','GOOG','GOOGL','META','BRK.B','LLY','TSLA','AVGO','WMT','JPM','UNH','V','XOM','MA','PG','JNJ','COST','ORCL','HD','ABBV','KO','BAC','MRK','NFLX','CVX','ADBE','PEP','TMO','CRM','TMUS','AMD','LIN','ACN','MCD','ABT','PM','DHR','CSCO','IBM','WFC','TXN','VZ','GE','QCOM','AXP','NOW','INTU','AMGN','ISRG','NEE','PFE','GS','CAT','SPGI','RTX','DIS','MS','T','CMCSA','UNP','PGR','UBER','AMAT','LOW','SYK','LMT','TJX','HON','BLK','BKNG','ELV','REGN','COP','BSX','VRTX','PLD','NKE','CB','MDT','SCHW','ETN','C','MMC','ADP','PANW','AMT','UPS','ADI','BX','DE','KKR','SBUX','ANET','MDLZ','BA','CI','HCA','FI','GILD','BMY','SO','MU','KLAC','LRCX','ICE','MO','SHW','DUK','MCO','CL','ZTS','WM','GD','INTC','CTAS','EQIX','CME','TT','WELL','NOC','AON','PH','CMG','ABNB','ITW','MSI','APH','TDG','PNC','SNPS','CVS','ECL','PYPL','USB','MMM','FDX','TGT','CDNS','BDX','EOG','MCK','AJG','CSX','ORLY','RSG','MAR','CARR','PSA','AFL','DHI','APD','CRWD','ROP','NXPI','NEM','NSC','FCX','FTNT','SLB','TFC','EMR','GEV','AEP','ADSK','TRV','O','CEG','MPC','COF','WMB','OKE','PSX','AZO','GM','HLT','MET','SPG','SRE','CCI','KDP','ROST','BK','PCAR','MNST','KMB','LEN','ALL','DLR','OXY','D','PAYX','CPRT','GWW','AIG','KMI','CHTR','COR','URI','JCI','STZ','FIS','KVUE','TEL','MSCI','IQV','KHC','FICO','LHX','RCL','VLO','AMP','F','PCG','ACGL','GIS','HUM','NDAQ','PRU','HSY','MPWR','CMI','ODFL','MCHP','PEG','A','EW','HES','IDXX','FAST','VRSK','GEHC','EXC','CTVA','SYY','HWM','EA','AME','IT','CTSH','KR','YUM','CNC','EXR','PWR','EFX','OTIS','RMD','ED','DOW','VICI','XEL','IR','GRMN','GLW','CBRE','HIG','DFS','BKR','NUE','EIX','DD','HPQ','AVB','CSGP','IRM','FANG','TRGP','XYL','EL','MLM','LYB','VMC','LULU','WEC','WTW','ON','BRO','LVS','MRNA','PPG','TSCO','ROK','MTD','EBAY','BIIB','CDW','WAB','EQR','AWK','ADM','MTB','NVR','FITB','DAL','GPN','DXCM','K','AXON','CAH','TTWO','PHM','ANSS','VLTO','VTR','IFF','ETR','DVN','CHD','DTE','SBAC','VST','FE','FTV','HAL','KEYS','TYL','STT','DOV','BR','ES','STE','RJF','ROL','SMCI','PPL','NTAP','TSN','SW','TROW','HPE','DECK','WRB','AEE','MKC','CBOE','WY','FSLR','WST','BF.B','INVH','LYV','GDDY','COO','WDC','CINF','ZBH','CPAY','STX','HBAN','BBY','ATO','ARE','LDOS','CMS','RF','CLX','CCL','HUBB','TER','PTC','BAX','TDY','WAT','BALL','BLDR','OMC','ESS','HOLX','LH','SYF','GPC','MOH','EQT','CFG','MAA','DRI','FOXA','APTV','PFG','PKG','ULTA','J','WBD','CNP','LUV','DG','HRL','VRSN','FOX','NTRS','AVY','L','JBHT','EXPE','EXPD','DGX','STLD','ZBRA','MAS','CTRA','EG','IP','ALGN','FDS','TXT','NRG','AMCR','UAL','SWKS','GEN','CAG','KIM','DOC','CPB','NWS','PODD','LNT','NWSA','UHS','KEY','NI','IEX','MRO','SWK','DPZ','UDR','RVTY','SNA','DLTR','AKAM','PNR','CF','NDSN','BG','ENPH','EVRG','REG','VTRS','TRMB','POOL','CE','CPT','SJM','JNPR','DVA','KMX','JKHY','INCY','CHRW','HST','EPAM','BXP','ALLE','IPG','FFIV','JBL','TAP','SOLV','TFX','AES','EMN','TECH','AOS','CTLT','RL','MGM','LKQ','HII','BEN','PNW','AIZ','QRVO','FRT','MKTX','CRL','TPR','HAS','MHK','MTCH','GL','APA','ALB','PAYC','LW','BIO','DAY','HSIC','GNRC','WYNN','MOS','CZR','NCLH','WBA','FMC','BWA','AAL','IVZ','PARA','BBWI','ETSY']
tickers = ['HOLX','LH','SYF','GPC','MOH','EQT','CFG','MAA','DRI','FOXA','APTV','PFG','PKG','ULTA','J','WBD','CNP','LUV','DG','HRL','VRSN','FOX','NTRS','AVY','L','JBHT','EXPE','EXPD','DGX','STLD','ZBRA','MAS','CTRA','EG','IP','ALGN','FDS','TXT','NRG','AMCR','UAL','SWKS','GEN','CAG','KIM','DOC','CPB','NWS','PODD','LNT','NWSA','UHS','KEY','NI','IEX','MRO','SWK','DPZ','UDR','RVTY','SNA','DLTR','AKAM','PNR','CF','NDSN','BG','ENPH','EVRG','REG','VTRS','TRMB','POOL','CE','CPT','SJM','JNPR','DVA','KMX','JKHY','INCY','CHRW','HST','EPAM','BXP','ALLE','IPG','FFIV','JBL','TAP','SOLV','TFX','AES','EMN','TECH','AOS','CTLT','RL','MGM','LKQ','HII','BEN','PNW','AIZ','QRVO','FRT','MKTX','CRL','TPR','HAS','MHK','MTCH','GL','APA','ALB','PAYC','LW','BIO','DAY','HSIC','GNRC','WYNN','MOS','CZR','NCLH','WBA','FMC','BWA','AAL','IVZ','PARA','BBWI','ETSY']
tickers = sorted(tickers)

ticker_descs = []
with open('data/sec_company_tickers.json', 'r', encoding='utf-8') as f:
    all_comps = json.load(f)
def get_comp_from_ticker(ticker: str) -> str:
    return next(company_data['title'] for company_data in all_comps.values() if company_data['ticker'] == ticker)
for ticker in tickers:
    ticker_descs.append(f'[{ticker}] {get_comp_from_ticker(ticker)}')

factors = [
    'Earnings Performance',
    'Profitability',
    'Valuation Metrics',
    'Liquidity and Solvency',
    'Cash Flow Health',
    'Balance Sheet Strength',
    'Risk Factors',
    'Capital Allocation',
    'Segment Performance',
    'Recent Developments in Statements'
]
factors = [f'{i}. {f}' for i, f in enumerate(factors, start=1)]
filter = {'date': {'$gt': '2024-09-30'}, 'form_type': {'$in': ['8-K', '10-Q']}}

def compare_companies(compA: str, compB: str, factor: str, main_history: list) -> Generator|tuple:
    if compA == compB:
        raise ValueError('Please choose different companies to compare.')

    factor_parts = factor.split('.')
    factor_num = int(factor_parts[0].strip())
    factor = factor_parts[1].strip()
    question = load_prompt(f'prompt{factor_num}.txt')

    async def ask_questions_in_parallel():
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            tasks = [loop.run_in_executor(executor, ask_question, comp, question, {**filter}) for comp in [compA, compB]]
            return await asyncio.gather(*tasks)

    answers = asyncio.run(ask_questions_in_parallel())
    compA_answer = answers[0]
    compB_answer = answers[1]

    yield [ai_message(compA_answer)], [ai_message(compB_answer)], gr.update(value=[ai_message('Comparing...')], show_copy_button=False), main_history

    super_model = get_llm_sambanova()

    reco_prompt = PromptTemplate.from_template(template=load_prompt(f'recommend_prompt.txt'))
    reco_prompt = reco_prompt.invoke({'factor': factor, 'compA': compA, 'compB': compB, 'compA_answer': compA_answer, 'compB_answer': compB_answer})
    reco_prompt = reco_prompt.to_string()

    reco_answer = super_model.invoke(reco_prompt)
    reco_answer = reco_answer.content
    print(f'{reco_answer=}')
    main_history.append(ai_message('Summarizing...'))

    yield [ai_message(compA_answer)], [ai_message(compB_answer)], gr.update(value=[ai_message(reco_answer)], show_copy_button=True), main_history

    summary_prompt = PromptTemplate.from_template(template=load_prompt(f'summarize_prompt.txt'))
    summary_prompt = summary_prompt.invoke({'factor': factor.lower(), 'compA': compA, 'compB': compB, 'reco_answer': reco_answer})
    summary_prompt = summary_prompt.to_string()

    summary_answer = super_model.invoke(summary_prompt)
    summary_answer = summary_answer.content
    print(f'{summary_answer=}')
    main_history.pop() # remove "Summarizing..."
    main_history.append(ai_message(f'<b>Summary:</b> {summary_answer}'))

    yield [ai_message(compA_answer)], [ai_message(compB_answer)], [ai_message(reco_answer)], gr.update(value=main_history, show_copy_all_button=True)

def ai_message(content: str) -> ChatMessage:
    return ChatMessage('assistant', content)

def create_chatbot(label: str, autoscroll=False) -> gr.Chatbot:
    chatbot = gr.Chatbot(
        label=label,
        type='messages',
        height='35vh',
        show_copy_button=True,
        autoscroll=autoscroll
    )
    return chatbot

with gr.Blocks(title=TITLE, theme=gr.themes.Glass(), css='''
    footer {visibility: hidden}

    /* for dropdown icons: increase size, make colour stronger */
    .wrap-inner .icon-wrap svg {
        width: 48px !important;
        height: 48px !important;
        fill: #0077b6 !important;
    }

    /* remove clear button in chatbots */
    button[aria-label="Clear"] {
        display: none !important;
    }
            ''') as demo:
    gr.Markdown(f'<h1 style="text-align:center;">{TITLE}📈📉</h1>')

    with gr.Row():
        main_chatbot = create_chatbot('SvS bOT', True)
        main_chatbot.value = [ChatMessage('assistant', GREETING)]
        main_chatbot.show_copy_button = False

    main_state = gr.State(main_chatbot.value)

    with gr.Row(variant='panel'):
        compA_dropdown = gr.Dropdown(label=f'{COMP_A}', choices=zip(ticker_descs, tickers), interactive=True)
        compB_dropdown = gr.Dropdown(label=f'{COMP_B}', choices=zip(ticker_descs, tickers), interactive=True, value=tickers[1])
        factor_dropdown = gr.Dropdown(label=f'{INVESTMENT_FACTOR.title()}', choices=factors, interactive=True)
        with gr.Column():
            compare_btn = gr.Button('Compare📈📉')
            reset_btn = gr.Button('Download')

    with gr.Row(variant='panel'):
        compA_chatbot = create_chatbot(COMP_A)
        compB_chatbot = create_chatbot(COMP_B)
        reco_chatbot = create_chatbot('SvS Recommendation')

    compare_btn_state = gr.State(True)
    compare_btn.click(
        fn=lambda: (gr.update(value=[]), gr.update(value=[]), gr.update(value=[])),
        inputs=None,
        outputs=[compA_chatbot, compB_chatbot, reco_chatbot]
    ).then(
        fn=compare_companies,
        inputs=[compA_dropdown, compB_dropdown, factor_dropdown, main_state],
        outputs=[compA_chatbot, compB_chatbot, reco_chatbot, main_chatbot])

demo.launch(server_name='0.0.0.0')
