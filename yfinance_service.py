import yfinance as yf

def get_current_stock_prices(tickers: list[str]) -> list[float|None]:
    prices = []
    for ticker in tickers:
        last_quote = None
        try:
            ticker_yahoo = yf.Ticker(ticker)
            data = ticker_yahoo.history()
            last_quote: float = data['Close'].iloc[-1]
            last_quote = round(last_quote, 2) # uses "half to even" rounding (banker's rounding)
        except Exception as e:
            print(f'Error fetching data for {ticker=}: {e}')
        prices.append(last_quote)
    return prices

if __name__ == '__main__':
    tickers = ['NVDA', 'AMD']
    prices = get_current_stock_prices(tickers)
    print(list(zip(tickers, prices)))
