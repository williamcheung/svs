from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import gradio as gr
import os
import time

from concurrent.futures import ThreadPoolExecutor
from gradio import ChatMessage
from langchain_core.prompts import PromptTemplate
from typing import Generator

from generate_report import gen_report
from langchain_tidb_rag import ask_question, ticker_in_data_file
from llm import get_llm_sambanova
from tidb_vector_store import all_tickers
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
SUMMARY_LABEL = '<b>Summary:</b>'
UTF8_ENCODING = 'utf-8'

MIN_DATE = os.getenv('MIN_DATE')

tickers = all_tickers

ticker_descs = []
with open('data/sec_company_tickers.json', 'r', encoding=UTF8_ENCODING) as f:
    all_comps = json.load(f)
def get_comp_from_ticker(ticker: str) -> str:
    return next(company_data['title'] for company_data in all_comps.values() if company_data['ticker'] == ticker_in_data_file(ticker))

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
filter = {'date': {'$gt': MIN_DATE}, 'form_type': {'$in': ['8-K', '10-K', '10-Q']}}

nbsp = '\u00A0'

def compare_companies(compA: str, compB: str, factor: str, main_history: list) -> Generator:
    if compA == compB:
        raise ValueError('Please choose different companies to compare.')

    try:
        factor_parts = factor.split('.')
        factor_num = int(factor_parts[0].strip())
        factor = factor_parts[1].strip()
        question = load_prompt(f'prompt{factor_num}.txt')
    except:
        question = f"{load_prompt('custom_prompt.txt')} {factor}"

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
    main_history.append(ai_message(f'{SUMMARY_LABEL} {summary_answer}'))

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

num_buttons = -1 # will set later

def disable_buttons() -> list[dict]:
    return enable_buttons(False)

def enable_buttons(enable=True) -> list[dict]:
    return [gr.update(interactive=enable) for _ in range(num_buttons)]

def generate_report(compA: list[dict], compB: list[dict], reco: list[dict], main: list[dict]) -> dict:
    compA_content = _get_content(compA)
    compB_content = _get_content(compB)
    reco_content = _get_content(reco)
    main_content = _get_content(main).lstrip(SUMMARY_LABEL).strip()

    html = gen_report(compA_content, compB_content, reco_content, main_content)

    file_path = f'reports/svs_report_{time.time_ns()}.html'
    with open(file_path, 'w', encoding=UTF8_ENCODING) as f:
        f.write(html)

    return gr.update(value=file_path, visible=True)

def _get_content(messages: list[dict]) -> str:
    return [msg['content'] for msg in messages][-1]

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

    # layout

    with gr.Row():
        main_chatbot = create_chatbot('SvS bOT', True)
        main_chatbot.value = [ChatMessage('assistant', GREETING)]
        main_chatbot.show_copy_button = False

    main_state = gr.State(main_chatbot.value)

    with gr.Row(variant='panel'):
        compA_dropdown = gr.Dropdown(label=f'{COMP_A} {nbsp}{nbsp}{nbsp}ℹ️(type-ahead supported :)', choices=zip(ticker_descs, tickers), interactive=True)
        compB_dropdown = gr.Dropdown(label=f'{COMP_B}', choices=zip(ticker_descs, tickers), interactive=True, value=tickers[1])
        factor_dropdown = gr.Dropdown(label=f'{INVESTMENT_FACTOR.title()}', choices=factors, interactive=True)
        with gr.Column():
            compare_btn = gr.Button('Compare📈📉')
            save_btn = gr.Button('Create Report', interactive=False)

    with gr.Row():
        report_file = gr.File(label='Report', interactive=False, visible=False)

    with gr.Row(variant='panel'):
        compA_chatbot = create_chatbot(COMP_A)
        compB_chatbot = create_chatbot(COMP_B)
        reco_chatbot = create_chatbot('SvS Recommendation')

    with gr.Row(equal_height=True):
        custom_factor = gr.Text('AI strategy, AI features in products and services, AI adoption', show_label=False, container=False, interactive=True, max_lines=1, scale=7)
        custom_compare_bn = gr.Button('Enter your own X-factor(s) to compare', scale=3)

    gr.HTML(
        '''
        <div id='app-footer' style='text-align: center;'>
            Powered by <a href='https://community.sambanova.ai/t/supported-models/193' target='_blank'>Llama 3.1 405B free on SambaNova</a>
        </div>
        '''
    )

    # handlers

    buttons = [compare_btn, save_btn, custom_compare_bn]
    num_buttons = len(buttons)

    compare_btn.click(
        fn=lambda: (gr.update(value=[]), gr.update(value=[]), gr.update(value=[]), gr.update(value=None, visible=False)),
        inputs=None,
        outputs=[compA_chatbot, compB_chatbot, reco_chatbot, report_file]
    ).then(
        fn=disable_buttons,
        inputs=None,
        outputs=buttons
    ).then(
        fn=compare_companies,
        inputs=[compA_dropdown, compB_dropdown, factor_dropdown, main_state],
        outputs=[compA_chatbot, compB_chatbot, reco_chatbot, main_chatbot]
    ).then(
        fn=enable_buttons,
        inputs=None,
        outputs=buttons
    )

    custom_compare_bn.click(
        fn=lambda: (gr.update(value=[]), gr.update(value=[]), gr.update(value=[]), gr.update(value=None, visible=False)),
        inputs=None,
        outputs=[compA_chatbot, compB_chatbot, reco_chatbot, report_file]
    ).then(
        fn=disable_buttons,
        inputs=None,
        outputs=buttons
    ).then(
        fn=compare_companies,
        inputs=[compA_dropdown, compB_dropdown, custom_factor, main_state],
        outputs=[compA_chatbot, compB_chatbot, reco_chatbot, main_chatbot]
    ).then(
        fn=enable_buttons,
        inputs=None,
        outputs=buttons
    )

    save_btn.click(
        fn=disable_buttons,
        inputs=None,
        outputs=buttons
    ).then(
        fn=generate_report,
        inputs=[compA_chatbot, compB_chatbot, reco_chatbot, main_chatbot],
        outputs=report_file
    ).then(
        fn=enable_buttons,
        inputs=None,
        outputs=buttons
    ).then(
        fn=lambda: gr.update(interactive=False),
        inputs=None,
        outputs=save_btn
    )

demo.launch(server_name='0.0.0.0')
