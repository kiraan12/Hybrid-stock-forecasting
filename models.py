import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')

class LinearRegressionForecaster:
    """Linear Regression model for stock price forecasting"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _create_features(self, prices):
        """Create features for linear regression"""
        features = []
        targets = []
        
        for i in range(self.window_size, len(prices)):
            # Use previous window_size prices as features
            feature_window = prices[i-self.window_size:i]
            target = prices[i]
            
            # Additional features
            ma_short = np.mean(feature_window[-5:])  # 5-day moving average
            ma_long = np.mean(feature_window)  # window_size-day moving average
            price_change = feature_window[-1] - feature_window[0]
            volatility = np.std(feature_window)
            
            feature_vector = list(feature_window) + [ma_short, ma_long, price_change, volatility]
            features.append(feature_vector)
            targets.append(target)
            
        return np.array(features), np.array(targets)
    
    def fit(self, prices, dates=None):
        """Train the linear regression model"""
        if len(prices) < self.window_size + 1:
            raise ValueError(f"Need at least {self.window_size + 1} data points")
        
        X, y = self._create_features(prices)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create features")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.last_prices = prices[-self.window_size:]
        self.is_fitted = True
        
    def predict(self, n_steps):
        """Make predictions for n_steps ahead"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        current_prices = self.last_prices.copy()
        
        for _ in range(n_steps):
            # Create features for prediction
            ma_short = np.mean(current_prices[-5:])
            ma_long = np.mean(current_prices)
            price_change = current_prices[-1] - current_prices[0]
            volatility = np.std(current_prices)
            
            feature_vector = list(current_prices) + [ma_short, ma_long, price_change, volatility]
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Make prediction
            pred = self.model.predict(feature_vector_scaled)[0]
            predictions.append(pred)
            
            # Update current_prices for next prediction
            current_prices = np.append(current_prices[1:], pred)
        
        return np.array(predictions)
    
    def forecast(self, n_steps):
        """Alias for predict method"""
        return self.predict(n_steps)


class ARIMAForecaster:
    """ARIMA model for stock price forecasting"""
    
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        
    def _check_stationarity(self, prices):
        """Check if the time series is stationary"""
        result = adfuller(prices)
        return result[1] <= 0.05  # p-value <= 0.05 indicates stationarity
    
    def _find_optimal_order(self, prices, max_p=3, max_d=2, max_q=3):
        """Find optimal ARIMA order using AIC"""
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(prices, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def fit(self, prices, dates=None):
        """Train the ARIMA model"""
        if len(prices) < 10:
            raise ValueError("Need at least 10 data points for ARIMA")
        
        try:
            # Try to find optimal order
            self.order = self._find_optimal_order(prices)
        except:
            # Fall back to default order
            self.order = (1, 1, 1)
        
        try:
            # Fit ARIMA model
            self.model = ARIMA(prices, order=self.order)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
        except Exception as e:
            # Try with simpler order if fitting fails
            try:
                self.order = (1, 1, 0)
                self.model = ARIMA(prices, order=self.order)
                self.fitted_model = self.model.fit()
                self.is_fitted = True
            except:
                # Fall back to even simpler model
                self.order = (1, 0, 0)
                self.model = ARIMA(prices, order=self.order)
                self.fitted_model = self.model.fit()
                self.is_fitted = True
    
    def predict(self, n_steps):
        """Make predictions for n_steps ahead"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            forecast = self.fitted_model.forecast(steps=n_steps)
            return np.array(forecast)
        except Exception as e:
            # If forecasting fails, return simple linear extrapolation
            last_price = self.fitted_model.fittedvalues.iloc[-1]
            trend = np.mean(np.diff(self.fitted_model.fittedvalues.iloc[-5:]))
            predictions = [last_price + trend * (i + 1) for i in range(n_steps)]
            return np.array(predictions)
    
    def forecast(self, n_steps):
        """Alias for predict method"""
        return self.predict(n_steps)


class EnsembleForecaster:
    """Ensemble model combining multiple forecasters"""
    
    def __init__(self, models=None):
        if models is None:
            self.models = [
                LinearRegressionForecaster(),
                ARIMAForecaster()
            ]
        else:
            self.models = models
        self.weights = None
        self.is_fitted = False
    
    def fit(self, prices, dates=None):
        """Train all models in the ensemble"""
        self.fitted_models = []
        
        for model in self.models:
            try:
                model.fit(prices, dates)
                self.fitted_models.append(model)
            except Exception as e:
                print(f"Failed to fit model {type(model).__name__}: {str(e)}")
        
        if not self.fitted_models:
            raise ValueError("No models could be fitted successfully")
        
        # Equal weights for now
        self.weights = [1.0 / len(self.fitted_models)] * len(self.fitted_models)
        self.is_fitted = True
    
    def predict(self, n_steps):
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        all_predictions = []
        
        for model in self.fitted_models:
            try:
                pred = model.predict(n_steps)
                all_predictions.append(pred)
            except Exception as e:
                print(f"Model {type(model).__name__} failed to predict: {str(e)}")
        
        if not all_predictions:
            raise ValueError("No models could make predictions")
        
        # Weighted average of predictions
        ensemble_pred = np.zeros(n_steps)
        total_weight = 0
        
        for i, pred in enumerate(all_predictions):
            weight = self.weights[i] if i < len(self.weights) else 1.0
            ensemble_pred += weight * pred
            total_weight += weight
        
        return ensemble_pred / total_weight
    
    def forecast(self, n_steps):
        """Alias for predict method"""
        return self.predict(n_steps)
