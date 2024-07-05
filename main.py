import io
import os
import tempfile
import traceback
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI, HTTPException, Query
from matplotlib import pyplot as plt
from pydantic import BaseModel
import yfinance as yf
import ta
import pandas as pd
from fastapi.responses import FileResponse

app = FastAPI()


class StockRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str


class CompanyNameRequest(BaseModel):
    name: str


@app.post("/stock-data/")
def get_stock_data(request: StockRequest):
    try:
        data = yf.download(request.ticker, start=request.start_date, end=request.end_date)
        if data.empty:
            raise HTTPException(status_code=404, detail="Data not found for the given ticker and date range")

        # Calculate technical indicators
        data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
        data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

        # Clean the data
        data = data.fillna(0)  # Replace NaNs with 0
        data = data.replace([float('inf'), float('-inf')], 0)  # Replace inf/-inf with 0

        return data.tail().to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Analysis API"}


@app.post("/stock-analysis/")
def analyze_stock(ticker: str):
    try:
        # Fetch the last 2 years of data
        data = yf.download(ticker, period='2y')
        if data.empty:
            raise HTTPException(status_code=404, detail="Data not found for the given ticker")

        # Calculate technical indicators
        data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
        data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

        # Feature engineering
        data['Price_Range'] = data['High'] - data['Low']
        data['Price_Change'] = data['Close'].pct_change() * 100
        data = data.fillna(0)

        # Prepare the data for prediction
        features = data[['SMA_50', 'SMA_200', 'RSI', 'Price_Range', 'Price_Change']].values
        target = data['Close'].values

        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(features[:-1], target[:-1])

        # Make a prediction for the next day's closing price
        predicted_price = model.predict([features[-1]])[0]

        # Simple heuristic for buy/sell recommendation
        current_price = data['Close'].iloc[-1]
        buy_price = current_price * 0.98  # Example: 2% below current price
        sell_price = predicted_price * 1.02  # Example: 2% above predicted price
        risk_percent = 100 - (model.score(features[:-1], target[:-1]) * 100)
        if predicted_price < current_price:
            should_buy = "Dont Buy"
            analysis = {
                "verdict": should_buy,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "risk_percent": risk_percent
            }

            return analysis
        else:
            should_buy = "Should Buy"
            analysis = {
                "verdict": should_buy,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "risk_percent": risk_percent
            }

            # return analysis

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search-companies/")
def search_companies(request: CompanyNameRequest):
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv('EQUITY_L.csv')

        # Search for matching company names in the DataFrame
        matching_companies = df[df['NAME OF COMPANY'].str.contains(request.name, case=False)]

        if matching_companies.empty:
            raise HTTPException(status_code=404, detail="No matching companies found")

        # Convert DataFrame to list of dictionaries
        results = matching_companies.to_dict(orient='records')

        return results

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="CSV file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stock-plot/")
async def stock_plot(ticker: str, years: int = Query(0, title="Years", ge=0),
                     months: int = Query(0, title="Months", ge=0)):
    try:
        # Calculate start and end dates based on years and months
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=years * 365 + months * 30)

        # Fetch historical data
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")

        # Calculate technical indicators
        data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label='Close Price')
        plt.plot(data['SMA_50'], label='50-Day SMA')
        plt.plot(data['SMA_200'], label='200-Day SMA')
        plt.title(f'{ticker} Stock Price and Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name)
        plt.close()

        # Return the plot as an image response
        return FileResponse(temp_file.name, media_type='image/png')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # finally:
    #     if 'temp_file' in locals() and os.path.exists(temp_file.name):
    #         os.remove(temp_file.name)
