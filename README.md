# Stock vs. Stock

This is an AI agent powered by Llama 3.1 405B on SambaNova that compares any pair of S&P 500 stocks based on a top 10 investment factor and gives an investment recommendation. The comparison is based on RAG with the 2 companies' recent 2024 financial statements (8-K, 10-K, 10-Q) filed with the SEC, which are embedded using OpenAI as vectors in TiDB. Also, a summary of the whole analysis and an optional report are generated by Llama. The user can also enter their own custom investment factors to do the stock vs. stock comparison, such as comparing AI strategy.

## Tech Stack

The agent is built in Python 3.12 using the following technologies:

- Llama 3.1 405B on SambaNova
- OpenAI Embedding
- TiDB Vector
- LangChain
- Gradio

## Docker Deployment

- Copy the `.env.sample` file to a new file named `.env`.

- Edit the `.env` file and update the "xxx" and "yyy" environment variable settings with real values from: https://sambanova.ai and https://platform.openai.com

- For the TiDB values, sign up for the free tier at: https://tidbcloud.com/free-trial
    - After signing up, click "Create Cluster" and choose `Serverless` (5 GB free forever).
    - Create a table named `embedded_edgar_filings` to match the `.env`.
    - Set the table's vector dimension to `768` to match the `.env`.
    - To set `TIDB_DATABASE_URL1` in the `.env`, click the "Connect" button top right, then select `Connect With`|`SQLAlchemy` to get your URL for `mysqlclient`. Finally click `Download the CA cert` and put the file `isrgrootx1.pem` in the `ca_cert` folder.
    - Repeat the steps with a second cluster because one is not enough to store all the vectors and metadata for the S&P 500 financial statements for 2024.
    - Set `TIDB_DATABASE_URL2` in the `.env` to the URL for your second cluster.

- Save your changes to the `.env` file.

- Download the financial statements for the S&P 500 companies from the SEC EDGAR database: https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
- Chunk and embed each statement in text form and load to TiDB with metadata fields:
    - ticker (with any dot replaced by dash)
    - url
    - title
    - date (e.g. 2024-11-22)
    - form_type (8-K, 10-K, 10-Q)
    - chunk (1, 2, 3, ...)
- You will need to split the stocks between your 2 clusters. Put the stocks in cluster 1 in `config\cluster1_tickers.txt` and those in cluster 2 in `config\cluster2_tickers.txt`.
- Note for the embedding use OpenAI text-embedding-3-large at 768 dimensions per the `.env` setting.

- Build the Docker image, and finally run the Docker container based on the image:
```bash
docker build -t stock-vs-stock .
docker run --name stock-vs-stock_container -p 7860:7860 -t stock-vs-stock
```

Once you see the following logs in the Docker container console, the agent's chatbot interface is running and accessible:
```bash
Running on local URL:  http://0.0.0.0:7860

To create a public link, set `share=True` in `launch()`.
```

Access the chatbot interface on your host machine in a browser via: http://localhost:7860