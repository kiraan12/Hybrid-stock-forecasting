📊 Hybrid Stock Forecasting Web App

This project is a hybrid time series forecasting system built with Streamlit, combining statistical models (ARIMA) and machine learning models (Linear Regression) to predict future stock prices.
It fetches live stock market data from Yahoo Finance (yfinance), processes it with technical indicators, and provides interactive visualizations and forecasting insights.

🚀 Features

📈 Real-time Stock Data: Fetches historical stock prices using yfinance.

🔎 Technical Indicator Analysis: Computes moving averages, volatility, returns, and other metrics.

🧠 Hybrid Forecasting: Combines ARIMA and Linear Regression models for better predictions.

📊 Visualization Dashboard: Interactive charts built with plotly for price trends, forecasts, and performance.

📉 Outlier & Volatility Detection: Identifies abnormal price movements.

🧪 Metrics Evaluation: Displays performance metrics (RMSE, MAE, MAPE) for forecasting models.

🗂️ Project Structure
hybrid-stock-forecasting/
│
├── app.py               # Main Streamlit web app
├── models.py            # ARIMA and ML model classes
├── utils.py             # Utility functions (data loading, feature engineering, metrics)
├── .streamlit/          # Streamlit configuration (optional)
└── README.md            # Project documentation

🛠️ Installation & Setup
1. Clone the repository
git clone https://github.com/your-username/hybrid-stock-forecasting.git
cd hybrid-stock-forecasting

2. Create a virtual environment
python -m venv venv
source venv/bin/activate     # macOS/Linux
.\venv\Scripts\activate      # Windows

3. Install dependencies

Create a requirements.txt file with the following:

streamlit
numpy
pandas
plotly
scikit-learn
statsmodels
yfinance


Then install them:

pip install -r requirements.txt

▶️ Run the Application
streamlit run app.py


The app will start on http://localhost:8501.
Open it in your browser to interact with the dashboard.

💡 Usage

Enter a valid stock symbol (e.g., AAPL, TSLA, INFY.NS, RELIANCE.NS).

Select the forecasting horizon and parameters.

View real-time charts, metrics, and hybrid model predictions.

Analyze results with volatility, outlier, and return plots.

📊 Example Stock Symbols
Company	Symbol
Apple	AAPL
Microsoft	MSFT
Amazon	AMZN
Tesla	TSLA
Reliance Industries (India)	RELIANCE.NS
Infosys (India)	INFY.NS
📦 Tech Stack

Python – Data processing & modeling

Streamlit – Web app framework

yfinance – Stock market data

scikit-learn – Regression models

statsmodels – ARIMA forecasting

Plotly – Interactive charts

NumPy / Pandas – Data manipulation

📈 Future Improvements

🔮 Add LSTM / Prophet for deep learning-based forecasting

📉 Include portfolio optimization features

📊 Integrate live news sentiment analysis for better predictions

🤝 Contributing

Contributions are welcome!
Fork the repo, create a feature branch, and submit a pull request.

📜 License

This project is licensed under the MIT License — feel free to use and modify it.
