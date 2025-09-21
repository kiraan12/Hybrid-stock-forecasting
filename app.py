import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
from models import ARIMAForecaster, LinearRegressionForecaster
from utils import calculate_metrics, validate_stock_symbol

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Stock Price Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("📈 Stock Price Forecasting Dashboard")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("Configuration")

# Stock symbol input
stock_symbol = st.sidebar.text_input(
    "Enter Stock Symbol",
    value="AAPL",
    help="Enter a valid stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
).upper()

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365*2),
        max_value=datetime.now()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )

# Forecast parameters
st.sidebar.subheader("Forecast Settings")
forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=1,
    max_value=30,
    value=7,
    help="Number of days to forecast into the future"
)

model_choice = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["Linear Regression", "ARIMA"],
    help="Choose the machine learning model for forecasting"
)

# Training split
train_split = st.sidebar.slider(
    "Training Data Split (%)",
    min_value=70,
    max_value=95,
    value=80,
    help="Percentage of data used for training"
) / 100

# Load data button
if st.sidebar.button("🔄 Load Data & Generate Forecast", type="primary"):
    # Validate inputs
    if not stock_symbol:
        st.error("Please enter a valid stock symbol")
        st.stop()
    
    if start_date >= end_date:
        st.error("Start date must be before end date")
        st.stop()
    
    # Show loading spinner
    with st.spinner(f"Fetching data for {stock_symbol}..."):
        try:
            # Fetch stock data
            ticker = yf.Ticker(stock_symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                st.error(f"No data found for symbol {stock_symbol}. Please check the symbol and try again.")
                st.stop()
            
            # Store data in session state
            st.session_state.stock_data = data
            st.session_state.stock_symbol = stock_symbol
            st.session_state.forecast_days = forecast_days
            st.session_state.model_choice = model_choice
            st.session_state.train_split = train_split
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.stop()

# Main content
if 'stock_data' in st.session_state:
    data = st.session_state.stock_data
    symbol = st.session_state.stock_symbol
    
    # Display basic info
    col1, col2, col3, col4 = st.columns(4)
    
    latest_price = data['Close'].iloc[-1]
    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
    price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
    
    with col1:
        st.metric(
            "Current Price",
            f"${latest_price:.2f}",
            f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        st.metric("Data Points", len(data))
    
    with col3:
        st.metric("Date Range", f"{len(data)} days")
    
    with col4:
        st.metric("Volatility", f"{data['Close'].std():.2f}")
    
    st.markdown("---")
    
    # Historical data visualization
    st.subheader("📊 Historical Price Data")
    
    # Create interactive chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Closing Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title=f"{symbol} Historical Closing Prices",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators
    st.subheader("📈 Technical Analysis")
    
    # Calculate moving averages
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    
    # Volume chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='blue')))
        fig_ma.add_trace(go.Scatter(x=data.index, y=data['MA_20'], name='MA 20', line=dict(color='orange')))
        fig_ma.add_trace(go.Scatter(x=data.index, y=data['MA_50'], name='MA 50', line=dict(color='red')))
        fig_ma.update_layout(title="Moving Averages", height=400)
        st.plotly_chart(fig_ma, use_container_width=True)
    
    with col2:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'))
        fig_vol.update_layout(title="Trading Volume", height=400)
        st.plotly_chart(fig_vol, use_container_width=True)
    
    st.markdown("---")
    
    # Forecasting section
    st.subheader("🔮 Price Forecasting")
    
    with st.spinner("Training model and generating forecasts..."):
        try:
            # Prepare data for modeling
            prices = data['Close'].values
            dates = data.index
            
            # Split data
            split_idx = int(len(prices) * st.session_state.train_split)
            train_prices = prices[:split_idx]
            test_prices = prices[split_idx:]
            train_dates = dates[:split_idx]
            test_dates = dates[split_idx:]
            
            # Initialize model
            if st.session_state.model_choice == "ARIMA":
                model = ARIMAForecaster()
            else:
                model = LinearRegressionForecaster()
            
            # Train model
            model.fit(train_prices, train_dates)
            
            # Make predictions on test set
            test_predictions = model.predict(len(test_prices))
            
            # Generate future forecasts
            future_predictions = model.forecast(st.session_state.forecast_days)
            
            # Create future dates
            last_date = dates[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=st.session_state.forecast_days,
                freq='D'
            )
            
            # Calculate metrics
            if len(test_prices) > 0:
                metrics = calculate_metrics(test_prices, test_predictions)
            else:
                metrics = None
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Forecast visualization
                fig_forecast = go.Figure()
                
                # Historical data
                fig_forecast.add_trace(go.Scatter(
                    x=train_dates,
                    y=train_prices,
                    mode='lines',
                    name='Training Data',
                    line=dict(color='blue')
                ))
                
                if len(test_prices) > 0:
                    # Test data (actual)
                    fig_forecast.add_trace(go.Scatter(
                        x=test_dates,
                        y=test_prices,
                        mode='lines',
                        name='Actual (Test)',
                        line=dict(color='green')
                    ))
                    
                    # Test predictions
                    fig_forecast.add_trace(go.Scatter(
                        x=test_dates,
                        y=test_predictions,
                        mode='lines',
                        name='Predicted (Test)',
                        line=dict(color='orange', dash='dash')
                    ))
                
                # Future forecasts
                fig_forecast.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    mode='lines+markers',
                    name='Future Forecast',
                    line=dict(color='red', width=3),
                    marker=dict(size=8)
                ))
                
                fig_forecast.update_layout(
                    title=f"{symbol} Price Forecast using {st.session_state.model_choice}",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
            
            with col2:
                # Model performance metrics
                st.subheader("📊 Model Performance")
                
                if metrics:
                    st.metric("MAE", f"${metrics['mae']:.2f}")
                    st.metric("RMSE", f"${metrics['rmse']:.2f}")
                    st.metric("MAPE", f"{metrics['mape']:.2f}%")
                    
                    # Accuracy based on direction
                    accuracy = metrics.get('accuracy', 0)
                    st.metric("Direction Accuracy", f"{accuracy:.1f}%")
                else:
                    st.info("No test data available for validation metrics")
                
                st.subheader("🔮 Forecast Summary")
                
                # Future predictions summary
                if len(future_predictions) > 0:
                    current_price = prices[-1]
                    forecast_price = future_predictions[-1]
                    price_change = forecast_price - current_price
                    price_change_pct = (price_change / current_price) * 100
                    
                    st.metric(
                        f"Predicted Price ({st.session_state.forecast_days}d)",
                        f"${forecast_price:.2f}",
                        f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                    )
                    
                    # Forecast confidence
                    volatility = np.std(prices[-30:])  # Last 30 days volatility
                    if volatility < 5:
                        confidence = "High"
                        confidence_color = "green"
                    elif volatility < 10:
                        confidence = "Medium"
                        confidence_color = "orange"
                    else:
                        confidence = "Low"
                        confidence_color = "red"
                    
                    st.markdown(f"**Confidence:** :{confidence_color}[{confidence}]")
            
            # Forecast table
            st.subheader("📅 Detailed Forecast")
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': [f"${price:.2f}" for price in future_predictions],
                'Change from Current': [f"${price - current_price:+.2f}" for price in future_predictions],
                'Change (%)': [f"{((price - current_price) / current_price) * 100:+.2f}%" for price in future_predictions]
            })
            
            st.dataframe(forecast_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in forecasting: {str(e)}")
            st.error("Please try with different parameters or check your data.")

else:
    # Initial state - show instructions
    st.info("👆 Please configure your settings in the sidebar and click 'Load Data & Generate Forecast' to begin.")
    
    # Sample usage instructions
    st.subheader("📋 How to Use")
    st.markdown("""
    1. **Enter a stock symbol** (e.g., AAPL, GOOGL, MSFT, TSLA)
    2. **Select date range** for historical data
    3. **Choose forecast horizon** (1-30 days)
    4. **Select forecasting model** (Linear Regression or ARIMA)
    5. **Adjust training split** percentage
    6. **Click 'Load Data & Generate Forecast'**
    
    The dashboard will display:
    - Historical price charts with technical indicators
    - Model performance metrics
    - Future price predictions
    - Detailed forecast table
    """)
    
    st.subheader("🔧 Features")
    st.markdown("""
    - **Real-time data** fetching using Yahoo Finance
    - **Interactive charts** with Plotly visualizations
    - **Multiple ML models** (ARIMA and Linear Regression)
    - **Technical analysis** with moving averages
    - **Performance metrics** (MAE, RMSE, MAPE)
    - **Configurable parameters** for different trading strategies
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Stock Price Forecasting Dashboard | Data provided by Yahoo Finance</p>
        <p><small>Disclaimer: This is for educational purposes only. Not financial advice.</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
