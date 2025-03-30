# Cryptocurrency Price Forecasting Using GRU (Gated Recurrent Unit) Models

## Table of Contents
1. [Introduction](#introduction)
2. [Theoretical Foundations of GRU](#theoretical-foundations-of-gru)
3. [Cryptocurrency Market Specifics](#cryptocurrency-market-specifics)
4. [GRU Application Methodology](#gru-application-methodology)
5. [Data Preparation](#data-preparation)
6. [Building a GRU Model](#building-a-gru-model)
7. [Performance Evaluation and Optimization](#performance-evaluation-and-optimization)
8. [Practical Implementation](#practical-implementation)
9. [Comparison with Other Methods](#comparison-with-other-methods)
10. [Limitations and Future Developments](#limitations-and-future-developments)
11. [Conclusion](#conclusion)
12. [References](#references)

## Introduction

Cryptocurrency price forecasting represents a complex financial modeling task due to high volatility, lack of centralized regulation, and dependence on numerous factors—from technical aspects and market sentiment to macroeconomic events and regulatory changes. Traditional methods of financial time series forecasting often prove insufficiently effective for the cryptocurrency market, which motivates researchers and practitioners to turn to deep learning methods.

Among various deep neural network architectures, recurrent models, in particular, GRU (Gated Recurrent Unit) and LSTM (Long Short-Term Memory), have shown significant potential in processing sequential data. The GRU model, proposed by Cho et al. in 2014, represents a simplified but no less effective version of LSTM, requiring fewer computational resources while maintaining a comparable ability to capture long-term dependencies in time series.

This research focuses on applying GRU models for cryptocurrency price forecasting, examining both theoretical aspects of this approach and practical solutions using modern machine learning tools.

## Theoretical Foundations of GRU

### GRU Architecture

GRU (Gated Recurrent Unit) is a type of recurrent neural network (RNN) designed to solve the vanishing and exploding gradient problem in standard RNNs. GRU was proposed as a simplified alternative to LSTM, preserving its key advantages but with fewer parameters and, consequently, faster training.

The main components of GRU:

1. **Reset Gate** — controls which part of the previous state will be forgotten. Formally, this gate is computed as:

   $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$

   where:
   - $r_t$ — reset gate value at time $t$
   - $\sigma$ — sigmoid activation function
   - $W_r$ — reset gate weight matrix
   - $h_{t-1}$ — hidden state at the previous step
   - $x_t$ — input data at time $t$
   - $b_r$ — bias vector

2. **Update Gate** — determines what part of the information from the current input will be added to the hidden state. Formally:

   $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$

   where:
   - $z_t$ — update gate value at time $t$
   - Other notations are similar to those above

3. **Candidate for New State** — calculated based on the current input and the part of the previous state that wasn't reset:

   $\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)$

   where:
   - $\tilde{h}_t$ — candidate for new state
   - $\odot$ — element-wise multiplication operation (Hadamard product)
   - $W$ — weight matrix for the new state candidate
   - $b$ — bias vector for the new state candidate

4. **Final Hidden State** — a combination of the previous state and the candidate for a new state, weighted by update gates:

   $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$

   where:
   - $h_t$ — new hidden state at time $t$

### Comparison of GRU with LSTM

| Characteristic | GRU | LSTM |
|---------------|-----|------|
| Number of gates | 2 (reset and update) | 3 (input, forget, output) |
| Number of parameters | Fewer | More |
| Training speed | Faster | Slower |
| Ability to capture long-term dependencies | Good | Excellent |
| Memory usage | Economical | More demanding |
| Application for small datasets | Often preferable | May overfit |

GRU represents a more compact model with fewer parameters, making its training faster and more efficient, especially with limited computational resources or small datasets. At the same time, LSTM may show better results on large datasets and when it's necessary to capture very long-term dependencies.

### Advantages of GRU for Time Series Forecasting

1. **Ability to Capture Long-Term Dependencies**
   - GRU effectively handles the vanishing gradient problem
   - Capable of remembering significant events even if they occurred many steps ago

2. **Training Efficiency**
   - Fewer parameters accelerate convergence
   - Requires fewer computational resources compared to LSTM

3. **Noise Resilience**
   - Gate structure allows the model to filter irrelevant information
   - Important property for volatile cryptocurrency data

4. **Adaptability to Different Time Scales**
   - Ability to work with both short-term and long-term patterns
   - Possibility to adjust the architecture for different time horizons of forecasting

## Cryptocurrency Market Specifics

### Unique Characteristics of Cryptocurrency Data

1. **High Volatility**
   - Daily fluctuations of 10-20% or more are not uncommon
   - Significantly higher than the volatility of traditional financial instruments

2. **24/7 Trading**
   - No breaks or holidays
   - Continuous data flow that may contain important patterns

3. **Heterogeneity of Market Participants**
   - From individual traders to institutional investors and large miners
   - Different time horizons and strategies of participants

4. **Presence of "Whales"**
   - Large volumes of tokens are concentrated among a small number of holders
   - Individual transactions can significantly affect the market

5. **Dependence on News Background**
   - High sensitivity to regulatory news
   - Impact of influencer tweets and statements from major companies

6. **Technical Peculiarities**
   - Hash rate, mining difficulty, block rewards, and other technical parameters
   - Impact of hard forks and protocol updates

### Key Factors Influencing Cryptocurrency Prices

1. **Market Factors**
   - Supply and demand
   - Trading volumes
   - Liquidity
   - Market capitalization

2. **Macroeconomic Factors**
   - Inflation
   - Returns on traditional financial instruments
   - Global economic crises

3. **Regulatory Actions**
   - Bans and restrictions in different countries
   - Implementation of regulatory frameworks
   - Tax changes

4. **Technological Development**
   - Protocol updates
   - Scalability
   - Security and vulnerabilities

5. **Social Perception**
   - Public trust
   - Media mentions
   - Social media activity

6. **Inter-Market Interactions**
   - Correlation with other cryptocurrencies
   - Interaction with traditional financial markets
   - Correlation with commodity markets (gold, oil)

### Challenges in Cryptocurrency Market Forecasting

1. **Non-Determinism**
   - Multiple unknown and unpredictable factors
   - "Black swans" — rare but significant events

2. **Limited Data History**
   - Relatively young market with a short history
   - Rapidly changing market characteristics make historical data less relevant

3. **Data Non-Stationarity**
   - Statistical properties of time series change over time
   - Models trained on historical data can quickly become outdated

4. **Market Manipulations**
   - Pump and dump schemes
   - Spoofing and other manipulative practices

5. **Technical Limitations**
   - Delays in data acquisition
   - Price differences between exchanges
   - Liquidity issues

## GRU Application Methodology

### Defining the Forecasting Task

When forecasting cryptocurrency prices using GRU, it's necessary to clearly define the task, which can take various forms:

1. **By Forecast Type**:
   - **Regression**: Forecasting a specific price value
   - **Classification**: Predicting the direction of price movement (up/down)
   - **Probabilistic Forecasting**: Estimating the probability distribution of future prices

2. **By Time Horizon**:
   - **Short-term forecasting**: minutes, hours, days
   - **Medium-term forecasting**: weeks, months
   - **Long-term forecasting**: quarters, years

3. **By Input Data Type**:
   - **Univariate time series**: only historical prices
   - **Multivariate time series**: prices, volumes, technical analysis indicators
   - **Multimodal data**: prices + news, social media, blockchain metrics

### Data Selection and Preparation

For effective application of GRU, proper selection and preparation of data is important:

1. **Data Sources**:
   - Centralized exchanges (Binance, Coinbase, Kraken)
   - DEX (Uniswap, PancakeSwap)
   - Aggregators (CoinMarketCap, CoinGecko)
   - Blockchain explorers for on-chain metrics
   - News sources and social media APIs

2. **Choosing Time Intervals**:
   - High-frequency data (minute, second) for short-term trading
   - Daily data for medium-term forecasts
   - Weekly/monthly data for long-term trends

3. **Historical Data Volume**:
   - Sufficient to capture complete market cycles
   - Taking into account different market regimes (bull/bear market)
   - Not too old to avoid including irrelevant patterns

### GRU Model Workflow Stages

The process of applying GRU for cryptocurrency price forecasting can be divided into the following stages:

1. **Data Collection and Initial Processing**
   - Obtaining historical data on prices, volumes, and other metrics
   - Handling missing values
   - Detecting and processing outliers

2. **Exploratory Data Analysis (EDA)**
   - Time series visualization
   - Checking stationarity and autocorrelation
   - Analysis of distributions and relationships between variables

3. **Data Preparation for the Model**
   - Normalization or standardization
   - Creating time windows (sequences)
   - Splitting into training, validation, and test sets

4. **Model Architecture Design**
   - Determining the number of GRU layers
   - Choosing the number of neurons in each layer
   - Adding regularization and Dropout

5. **Model Training**
   - Choosing an optimizer and loss function
   - Determining the number of epochs and mini-batch size
   - Monitoring the training process to prevent overfitting

6. **Model Evaluation and Optimization**
   - Analysis of metrics on the validation set
   - Fine-tuning hyperparameters
   - Cross-validation to increase result reliability

7. **Testing and Interpretation**
   - Evaluation on the test set
   - Interpretation of results
   - Analysis of model errors

## Data Preparation

### Preprocessing

Data quality and preparation are critically important for GRU model effectiveness. The main preprocessing steps include:

1. **Handling Missing Values**
   - Filling using interpolation
   - Using the last known value (forward fill)
   - Removing rows with missing values (if there are few)

2. **Handling Outliers**
   - Identifying outliers using statistical methods (Z-score, IQR)
   - Winsorization (limiting extreme values)
   - Logarithmic transformation to smooth the distribution

3. **Data Aggregation**
   - Converting data to the desired timeframe (minute, hourly, daily)
   - Computing OHLCV (Open, High, Low, Close, Volume) for each interval
   - Aggregating additional metrics

4. **Checking and Ensuring Stationarity**
   - Dickey-Fuller test for stationarity check
   - Differencing time series
   - Transformations to remove trend and seasonality

### Feature Engineering

For improving the predictive capability of the GRU model, it's important to create relevant features:

1. **Technical Indicators**
   - Moving averages (SMA, EMA)
   - Trend indicators (MACD, ADX)
   - Oscillators (RSI, Stochastic)
   - Volatility indicators (Bollinger Bands, ATR)

2. **Time Structure-Based Features**
   - Lagged variables (previous price values)
   - Differences between consecutive values
   - Percentage changes
   - Time components (hour of day, day of week, month)

3. **Volume and Market Metrics**
   - Price-to-volume ratio
   - Market depth indicators
   - Liquidity metrics
   - On-chain metrics (for blockchain assets)

4. **External Data**
   - Market indices
   - Social media sentiment analysis
   - Number of mentions in news
   - Macroeconomic indicators

### Normalization and Scaling

GRU, like other neural networks, is sensitive to the scale of input data. Therefore, normalization is necessary:

1. **Normalization Methods**
   - Min-Max normalization (scaling to a range [0, 1] or [-1, 1])
   - Standardization (Z-score normalization)
   - Robust scaling (using quantiles)

2. **Time Series Normalization Peculiarities**
   - Preventing data leakage (using only the training set to calculate normalization parameters)
   - Preserving the time structure of data
   - Sequential application of transformations over time

3. **Implementation Examples**

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# For Min-Max normalization
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data)

# For standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
```

### Creating Sequences for Training

For working with RNNs, including GRU, it's necessary to transform data into sequence format:

1. **Sliding Window Approach**
   - Defining the input window size (number of past observations)
   - Defining the forecast horizon (number of future observations)
   - Creating "input-output" pairs for training

2. **Example of Sequence Formation**

```python
def create_sequences(data, seq_length, horizon=1):
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length + horizon - 1])
    return np.array(X), np.array(y)

# Example usage
sequence_length = 60  # 60 time steps for prediction
X, y = create_sequences(scaled_data, sequence_length)
```

3. **Data Splitting**
   - Splitting into training, validation, and test sets, taking into account the time structure
   - Typical ratio: 70% training, 15% validation, 15% test

```python
# Example of data splitting
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
```

## Building a GRU Model

### Model Architecture

Designing a GRU model architecture involves several key decisions:

1. **Number of GRU Layers**
   - Single-layer models — for simple tasks and small datasets
   - Two or three layers — for more complex tasks
   - Deeper architectures — for capturing complex patterns, but with the risk of overfitting

2. **Number of Neurons in Each Layer**
   - Typical values: 32, 64, 128, 256
   - More neurons — higher learning capacity, but higher risk of overfitting
   - Fewer neurons — lower risk of overfitting, but possible underfitting

3. **Organization of Multi-Layer GRU**
   - `return_sequences=True` for all layers except the last one
   - Possibility of using Bidirectional GRU to account for future context
   - Residual connections for deep networks

4. **Output Layers**
   - Dense layers for transforming GRU output into the final forecast
   - Activation functions depending on the task (linear for regression, sigmoid/softmax for classification)

5. **Example of Basic Model Architecture**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

# Example of GRU model architecture
model = Sequential([
    GRU(units=128, activation='tanh', return_sequences=True, input_shape=(sequence_length, features)),
    Dropout(0.2),
    GRU(units=64, activation='tanh'),
    Dropout(0.2),
    Dense(units=16, activation='relu'),
    Dense(units=1)  # Output layer for regression
])
```

### Regularization and Overfitting Prevention

Deep learning models, including GRU, are prone to overfitting, especially on volatile cryptocurrency data:

1. **Regularization Methods**
   - Dropout — random neuron deactivation during training
   - L1/L2 regularization — adding a penalty for large weights
   - Early stopping — stopping training when no improvement on the validation set
   - Batch normalization — normalizing activations within the network

2. **Optimal Values for Cryptocurrency Data**
   - Dropout: 0.2-0.5 depending on model size
   - L2 regularization: 1e-4 to 1e-2
   - Patience for early stopping: 10-30 epochs

3. **Example of Regularization Implementation**

```python
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Example of regularization usage
model = Sequential([
    GRU(128, return_sequences=True, kernel_regularizer=l2(1e-4), input_shape=(sequence_length, features)),
    Dropout(0.3),
    GRU(64, kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer=l2(1e-4)),
    Dense(1)
])

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)
```

### Hyperparameter Selection

Hyperparameter tuning is critically important for GRU model effectiveness:

1. **Key Hyperparameters**
   - Input sequence length
   - Batch size
   - Learning rate
   - Activation functions
   - Optimizer

2. **Hyperparameter Selection Strategies**
   - Grid Search
   - Random Search
   - Bayesian Optimization
   - Evolutionary Algorithms

3. **Recommendations for Cryptocurrency Data**
   - Sequence length: from several days to several months depending on the timeframe
   - Batch size: 32-128 for balance between speed and stability
   - Optimizer: Adam with initial learning rate of 0.001
   - Learning rate schedule: reducing learning rate at validation error plateau

4. **Example of Hyperparameter Tuning**

```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Optimizer with learning rate setting
optimizer = Adam(learning_rate=0.001)

# Learning rate reduction at plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# Model compilation
model.compile(
    optimizer=optimizer,
    loss='mean_squared_error',
    metrics=['mae']
)

# Model training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
```

## Performance Evaluation and Optimization

### Forecast Quality Evaluation Metrics

Various metrics are used to evaluate the effectiveness of GRU models in cryptocurrency price forecasting:

1. **Regression Task Metrics**
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
   - RMSE (Root Mean Squared Error)
   - MAPE (Mean Absolute Percentage Error)
   - R² (Coefficient of Determination)

2. **Classification Metrics for Price Movement Direction**
   - Accuracy — proportion of correct predictions
   - Precision — proportion of true positives among all positive predictions
   - Recall — proportion of true positives among all actual positive cases
   - F1-score — harmonic mean of precision and recall
   - ROC AUC — area under the ROC curve

3. **Financial Metrics**
   - Profit & Loss (P&L) — profit and loss from trading strategy based on predictions
   - Sharpe Ratio — ratio of excess return to volatility
   - Maximum Drawdown — maximum observed loss from a peak
   - Win Rate — percentage of successful trades

4. **Example of Metric Calculation**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Getting predictions
y_pred = model.predict(X_test)

# Inverse transformation for comparison in original scale
y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_orig = scaler.inverse_transform(y_pred)

# Calculating metrics
mae = mean_absolute_error(y_test_orig, y_pred_orig)
mse = mean_squared_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_orig, y_pred_orig)

print(f"MAE: ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R² Score: {r2:.4f}")
```

### Results Interpretation

For effective use of GRU models, it's important to correctly interpret the results:

1. **Forecast Error Analysis**
   - Visualization of predictions and actual values
   - Residual analysis
   - Identifying systematic errors (bias)

2. **Model Behavior Analysis in Various Market Conditions**
   - Accuracy assessment in periods of high/low volatility
   - Model behavior during sharp market changes
   - Ability to predict trend reversal points

3. **Comparative Analysis**
   - Comparison with naive models (e.g., "tomorrow will be like today" forecast)
   - Comparison with traditional statistical methods (ARIMA, GARCH)
   - Comparison with other neural network architectures

4. **Result Visualization**

```python
import matplotlib.pyplot as plt

# Prediction visualization
plt.figure(figsize=(15, 7))
plt.plot(y_test_orig, label='Actual Prices')
plt.plot(y_pred_orig, label='Predicted Prices')
plt.title('BTC Price Prediction using GRU')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Residual analysis
residuals = y_test_orig - y_pred_orig
plt.figure(figsize=(15, 7))
plt.plot(residuals)
plt.title('Residuals (Prediction Errors)')
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Time')
plt.ylabel('Error (USD)')
plt.grid(True)
plt.show()
```

### Model Optimization and Fine-Tuning

Various optimization methods can be applied to improve GRU model performance:

1. **Automatic Hyperparameter Tuning**
   - Using frameworks for automatic optimization (Keras Tuner, Optuna, Hyperopt)
   - Defining search space for key hyperparameters
   - Evaluating each configuration using cross-validation

2. **Ensemble Methods**
   - Training several GRU models with different initializations
   - Combining predictions from different models (averaging, weighted averaging, stacking)
   - Creating heterogeneous ensembles (GRU + LSTM + CNN)

3. **Adaptive Learning**
   - Dynamically changing learning parameters depending on data characteristics
   - Retraining the model when market regime changes
   - Using online learning to update the model in real-time

4. **Example of Hyperparameter Optimization Implementation**

```python
from kerastuner.tuners import RandomSearch

def build_model(hp):
    model = Sequential()
    
    # First GRU layer tuning
    units = hp.Int('units_1', min_value=32, max_value=256, step=32)
    model.add(GRU(units=units, 
                  return_sequences=True, 
                  input_shape=(sequence_length, features)))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Second GRU layer tuning
    units = hp.Int('units_2', min_value=16, max_value=128, step=16)
    model.add(GRU(units=units))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Output layer
    model.add(Dense(1))
    
    # Optimizer tuning
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

# Creating tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=50,
    executions_per_trial=2,
    directory='tuning_results',
    project_name='crypto_gru'
)

# Searching for optimal hyperparameters
tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
)

# Getting the best model
best_model = tuner.get_best_models(num_models=1)[0]
```

## Practical Implementation

### Complete Python Pipeline Example

Below is a practical example of the complete process of cryptocurrency price forecasting using GRU:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Data Collection
# Assuming we have a DataFrame with historical data
# df = pd.read_csv('btc_historical.csv')

# For example, let's create synthetic data
def generate_synthetic_data(n=1000):
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n)
    trend = np.linspace(5000, 50000, n) + np.random.normal(0, 2000, n)
    seasonality = 2000 * np.sin(np.linspace(0, 10*np.pi, n))
    noise = np.random.normal(0, 3000, n)
    price = trend + seasonality + noise
    price = np.maximum(price, 1000)  # Price cannot be below 1000
    
    volume = np.random.lognormal(15, 1, n)
    df = pd.DataFrame({
        'date': dates,
        'close': price,
        'volume': volume
    })
    return df

df = generate_synthetic_data(1000)
df.set_index('date', inplace=True)

# 2. Data Preprocessing
# Creating additional features
df['return'] = df['close'].pct_change()
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df['volatility'] = df['log_return'].rolling(window=20).std()
df['ma_7'] = df['close'].rolling(window=7).mean()
df['ma_30'] = df['close'].rolling(window=30).mean()
df['volume_ma_7'] = df['volume'].rolling(window=7).mean()

# Removing rows with NaN
df.dropna(inplace=True)

# Selecting features
features = ['close', 'volume', 'return', 'volatility', 'ma_7', 'ma_30', 'volume_ma_7']
data = df[features].values

# 3. Data Normalization
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data)

# 4. Creating Sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Predicting only the closing price
    return np.array(X), np.array(y)

sequence_length = 60  # 60 days for prediction
X, y = create_sequences(scaled_data, sequence_length)

# 5. Data Splitting
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# 6. Building GRU Model
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(sequence_length, len(features))),
    Dropout(0.3),
    GRU(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

# 7. Model Compilation
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# 8. Callbacks for Training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# 9. Model Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 10. Visualizing the Training Process
plt.figure(figsize=(15, 7))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 11. Evaluating the Model on Test Data
y_pred = model.predict(X_test)

# Transforming back to original scale
y_test_scaled = np.zeros((len(y_test), len(features)))
y_test_scaled[:, 0] = y_test
y_test_orig = scaler.inverse_transform(y_test_scaled)[:, 0]

y_pred_scaled = np.zeros((len(y_pred), len(features)))
y_pred_scaled[:, 0] = y_pred.flatten()
y_pred_orig = scaler.inverse_transform(y_pred_scaled)[:, 0]

# 12. Calculating Metrics
mae = mean_absolute_error(y_test_orig, y_pred_orig)
mse = mean_squared_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_orig, y_pred_orig)

print(f"MAE: ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# 13. Visualizing the Results
plt.figure(figsize=(15, 7))
plt.plot(df.index[train_size+val_size+sequence_length:], y_test_orig, label='Actual BTC Price')
plt.plot(df.index[train_size+val_size+sequence_length:], y_pred_orig, label='Predicted BTC Price')
plt.title('Bitcoin Price Prediction using GRU')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# 14. Forecasting the Future
def predict_next_n_days(model, last_sequence, n_days=30):
    predictions = []
    curr_sequence = last_sequence.copy()
    
    for _ in range(n_days):
        # Predicting the next day
        next_pred = model.predict(curr_sequence.reshape(1, sequence_length, len(features)))
        
        # Adding the prediction to the list
        predictions.append(next_pred[0, 0])
        
        # Updating the sequence for the next prediction
        # Creating a new data point by duplicating the last row
        new_point = curr_sequence[-1].copy()
        new_point[0] = next_pred[0, 0]  # Updating only the price
        
        # Updating the sequence by removing the first point and adding the new one
        curr_sequence = np.vstack([curr_sequence[1:], new_point])
    
    return np.array(predictions)

# Getting the last sequence from test data
last_sequence = X_test[-1]

# Predicting the next 30 days
future_preds_scaled = predict_next_n_days(model, last_sequence, n_days=30)

# Transforming to original scale
future_preds_full = np.zeros((len(future_preds_scaled), len(features)))
future_preds_full[:, 0] = future_preds_scaled
future_preds = scaler.inverse_transform(future_preds_full)[:, 0]

# Creating dates for future predictions
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

# Visualizing the future forecast
plt.figure(figsize=(15, 7))
plt.plot(df.index[-100:], df['close'][-100:], label='Historical BTC Price')
plt.plot(future_dates, future_preds, label='Future BTC Price Prediction', color='red')
plt.title('Bitcoin Price Prediction for Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
```

### Working with Real Cryptocurrency Data

Various APIs can be used for working with real cryptocurrency data. Below is an example of obtaining data using CCXT (a library for working with cryptocurrency exchange APIs):

```python
import ccxt
import pandas as pd
from datetime import datetime

# Exchange API initialization
exchange = ccxt.binance({
    'enableRateLimit': True,  # Important for complying with API limitations
})

# Getting OHLCV data
def fetch_ohlcv_data(symbol, timeframe='1d', limit=1000):
    try:
        # Getting OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Converting to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Converting timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Example of getting data
btc_data = fetch_ohlcv_data('BTC/USDT', timeframe='1d', limit=1000)

# Displaying the first 5 rows of data
print(btc_data.head())
```

For working with market sentiment data, you can use social media APIs or specialized services:

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

# Example function for getting sentiment data from the CryptoFear & Greed Index API
def fetch_fear_greed_index(limit=30):
    url = f"https://api.alternative.me/fng/?limit={limit}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Converting to DataFrame
        df = pd.DataFrame(data['data'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['value'] = df['value'].astype(int)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)  # Sorting by date
        
        return df
    
    except Exception as e:
        print(f"Error fetching Fear & Greed Index: {e}")
        return None

# Getting data
fear_greed_data = fetch_fear_greed_index(limit=100)

# Displaying the first 5 rows of data
print(fear_greed_data.head())
```

For collecting blockchain data, you can use specialized APIs:

```python
import requests
import pandas as pd

# Example function for getting Bitcoin on-chain data through Blockchain.info API
def fetch_blockchain_data(days=30):
    # Endpoint for getting network activity
    url = f"https://api.blockchain.info/charts/n-transactions?timespan={days}days&format=json"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Converting to DataFrame
        values = data['values']
        df = pd.DataFrame(values)
        df['x'] = pd.to_datetime(df['x'], unit='s')
        df.rename(columns={'x': 'timestamp', 'y': 'transactions'}, inplace=True)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    except Exception as e:
        print(f"Error fetching blockchain data: {e}")
        return None

# Getting data
blockchain_data = fetch_blockchain_data(days=60)

# Displaying the first 5 rows of data
print(blockchain_data.head())
```

### Model Implementation and Deployment

After developing the model, it needs to be deployed for practical use:

1. **Saving and Loading the Model**

```python
# Saving the trained model
model.save('crypto_gru_model.h5')

# Loading the saved model
from tensorflow.keras.models import load_model
loaded_model = load_model('crypto_gru_model.h5')
```

2. **Deployment Using Flask API**

```python
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd

app = Flask(__name__)

# Loading model and scaler
model = load_model('crypto_gru_model.h5')
scaler = joblib.load('scaler.save')
sequence_length = 60
feature_count = 7  # Number of features

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Getting data from request
        data = request.get_json()
        input_sequence = np.array(data['sequence'])
        
        # Checking input data dimensions
        if input_sequence.shape != (sequence_length, feature_count):
            return jsonify({'error': f'Expected input shape: ({sequence_length}, {feature_count})'}), 400
        
        # Transforming input data
        input_sequence = input_sequence.reshape(1, sequence_length, feature_count)
        
        # Getting prediction
        prediction = model.predict(input_sequence)
        
        # Inverse scale transformation
        pred_scaled = np.zeros((1, feature_count))
        pred_scaled[0, 0] = prediction[0, 0]
        pred_orig = scaler.inverse_transform(pred_scaled)[0, 0]
        
        # Returning the result
        return jsonify({
            'prediction': float(pred_orig),
            'timestamp': pd.Timestamp.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

3. **Automated Trading System**

```python
import ccxt
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import load_model
import joblib

# Loading model and scaler
model = load_model('crypto_gru_model.h5')
scaler = joblib.load('scaler.save')
sequence_length = 60
feature_count = 7

# Exchange API initialization
exchange = ccxt.binance({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
    'enableRateLimit': True,
})

# Function for getting and preparing data
def prepare_data(symbol, timeframe='1h', limit=100):
    # Getting OHLCV data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Creating features
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_return'].rolling(window=20).std()
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_30'] = df['close'].rolling(window=30).mean()
    df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
    
    # Removing NaN
    df.dropna(inplace=True)
    
    # Selecting the last sequence_length rows
    features = ['close', 'volume', 'return', 'volatility', 'ma_7', 'ma_30', 'volume_ma_7']
    data = df[features].values[-sequence_length:]
    
    # Normalization
    scaled_data = scaler.transform(data)
    
    return scaled_data, df

# Trading function
def trading_bot(symbol, timeframe='1h', threshold=0.01):
    while True:
        try:
            print(f"Getting data for {symbol}...")
            scaled_data, df = prepare_data(symbol, timeframe)
            
            # Getting current price
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Prediction
            prediction = model.predict(scaled_data.reshape(1, sequence_length, feature_count))
            
            # Inverse scale transformation
            pred_scaled = np.zeros((1, feature_count))
            pred_scaled[0, 0] = prediction[0, 0]
            predicted_price = scaler.inverse_transform(pred_scaled)[0, 0]
            
            # Calculating expected change
            expected_change = (predicted_price - current_price) / current_price
            
            print(f"Current price: {current_price}")
            print(f"Predicted price: {predicted_price}")
            print(f"Expected change: {expected_change:.2%}")
            
            # Trading logic
            if expected_change > threshold:
                print("Signal: BUY")
                # Code for buying here
                # exchange.create_market_buy_order(symbol, amount)
            elif expected_change < -threshold:
                print("Signal: SELL")
                # Code for selling here
                # exchange.create_market_sell_order(symbol, amount)
            else:
                print("Signal: HOLD")
            
            # Pause before next iteration
            print(f"Waiting for next iteration...")
            time.sleep(3600)  # 1 hour
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)  # Wait a minute in case of error

# Starting the bot
trading_bot('BTC/USDT', timeframe='1h', threshold=0.01)
```

## Comparison with Other Methods

### Comparison of GRU with Traditional Methods

For a full understanding of GRU effectiveness, it's necessary to compare it with traditional forecasting methods:

1. **Statistical Methods**
   - ARIMA (AutoRegressive Integrated Moving Average)
   - GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)
   - Exponential Smoothing
   - VAR (Vector AutoRegression)

2. **Comparative Analysis of GRU and ARIMA**

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Assuming we have cryptocurrency data in DataFrame df
# with 'close' column for closing prices

# Splitting data for training and testing
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

# Training ARIMA model
def train_arima(train_data, p=5, d=1, q=0):
    model = ARIMA(train_data['close'], order=(p, d, q))
    model_fit = model.fit()
    return model_fit

# Training the model
arima_model = train_arima(train_data)

# Forecasting with ARIMA
forecast = arima_model.forecast(steps=len(test_data))
forecast_df = pd.DataFrame(forecast, index=test_data.index, columns=['forecast'])

# Comparing with actual data
arima_mae = mean_absolute_error(test_data['close'], forecast_df['forecast'])
arima_rmse = np.sqrt(mean_squared_error(test_data['close'], forecast_df['forecast']))
arima_r2 = r2_score(test_data['close'], forecast_df['forecast'])

print(f"ARIMA MAE: ${arima_mae:.2f}")
print(f"ARIMA RMSE: ${arima_rmse:.2f}")
print(f"ARIMA R² Score: {arima_r2:.4f}")

# Visualizing the comparison
plt.figure(figsize=(15, 7))
plt.plot(test_data.index, test_data['close'], label='Actual BTC Price')
plt.plot(forecast_df.index, forecast_df['forecast'], label='ARIMA Forecast', color='red')
plt.plot(test_data.index, y_pred_orig, label='GRU Forecast', color='green')
plt.title('BTC Price Prediction: ARIMA vs GRU')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Comparing metrics
metrics_df = pd.DataFrame({
    'ARIMA': [arima_mae, arima_rmse, arima_r2],
    'GRU': [mae, rmse, r2]
}, index=['MAE', 'RMSE', 'R²'])

print(metrics_df)
```

### Comparison of GRU with Other Neural Network Architectures

It's useful to compare GRU with other neural network architectures:

1. **LSTM**
   - Architecture similar to GRU, but with an additional "forget gate" mechanism
   - Theoretically can better remember long-term dependencies
   - Requires more parameters and computational resources

2. **1D CNN (Convolutional Neural Networks for Time Series)**
   - Effective for extracting local patterns
   - Faster training than recurrent networks
   - May miss long-term dependencies

3. **Transformers**
   - Modern architecture with attention mechanism
   - Parallel sequence processing
   - Effective for capturing dependencies between different parts of a time series

4. **Hybrid Models**
   - CNN-GRU: CNN for feature extraction, GRU for sequence processing
   - GRU-Attention: GRU with attention mechanism to focus on important parts of the sequence
   - LSTM-GRU: combining advantages of both architectures

5. **Example of GRU and LSTM Comparison**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Creating an LSTM model similar to the GRU model
def create_lstm_model(sequence_length, features):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, features)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Training LSTM model
lstm_model = create_lstm_model(sequence_length, len(features))
lstm_history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluating LSTM model
lstm_pred = lstm_model.predict(X_test)

# Inverse transformation
lstm_pred_scaled = np.zeros((len(lstm_pred), len(features)))
lstm_pred_scaled[:, 0] = lstm_pred.flatten()
lstm_pred_orig = scaler.inverse_transform(lstm_pred_scaled)[:, 0]

# Calculating metrics
lstm_mae = mean_absolute_error(y_test_orig, lstm_pred_orig)
lstm_mse = mean_squared_error(y_test_orig, lstm_pred_orig)
lstm_rmse = np.sqrt(lstm_mse)
lstm_r2 = r2_score(y_test_orig, lstm_pred_orig)

print(f"LSTM MAE: ${lstm_mae:.2f}")
print(f"LSTM RMSE: ${lstm_rmse:.2f}")
print(f"LSTM R² Score: {lstm_r2:.4f}")

# Comparing metrics
metrics_df = pd.DataFrame({
    'GRU': [mae, rmse, r2],
    'LSTM': [lstm_mae, lstm_rmse, lstm_r2]
}, index=['MAE', 'RMSE', 'R²'])

print(metrics_df)

# Visualizing the comparison
plt.figure(figsize=(15, 7))
plt.plot(df.index[train_size+val_size+sequence_length:], y_test_orig, label='Actual BTC Price')
plt.plot(df.index[train_size+val_size+sequence_length:], y_pred_orig, label='GRU Prediction', alpha=0.7)
plt.plot(df.index[train_size+val_size+sequence_length:], lstm_pred_orig, label='LSTM Prediction', alpha=0.7)
plt.title('BTC Price Prediction: GRU vs LSTM')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
```

### Influence of Various Factors on Forecast Accuracy

Many factors can affect the accuracy of forecasting using GRU:

1. **Input Variable Selection**
   - Historical prices only vs. multivariate input data
   - Influence of technical indicators
   - Role of trading volumes and market liquidity

2. **Forecast Time Horizon**
   - Short-term vs. long-term forecasts
   - Error scaling with increasing horizon

3. **Data Preprocessing**
   - Influence of different scaling methods
   - Role of outlier detection and removal
   - Influence of data transformation (logarithmic, differencing)

4. **Model Hyperparameters**
   - Input sequence size
   - Number of layers and neurons
   - Influence of regularization and dropout

## Limitations and Future Developments

### Fundamental Approach Limitations

Despite GRU effectiveness, there are fundamental limitations to this approach:

1. **Cryptocurrency Market Uncertainty**
   - High sensitivity to unpredictable events
   - Influence of "black swans" (extremely rare but significant events)
   - Limited predictability in principle

2. **Overfitting Problem**
   - Risk of "memorizing" historical patterns without identifying general regularities
   - Difficulty generalizing to new market regimes
   - "Forgetting curve" — accuracy decline over time

3. **Computational Complexity**
   - Resource demands for deep models
   - Complexity of real-time training
   - Balance between model complexity and prediction speed

4. **Interpretability**
   - Difficulty explaining reasons for specific predictions
   - "Black box" nature of deep neural networks
   - Difficulty identifying cause-and-effect relationships

### Promising Development Directions

There are several promising directions for improving forecasting using GRU:

1. **Integration with Other Data Types**
   - Including social media and news feed data
   - Analysis of on-chain metrics and transaction graphs
   - Accounting for macroeconomic indicators and correlations with traditional markets

2. **Hybrid and Multimodal Approaches**
   - Combining GRU with natural language processing models for news analysis
   - Integration with graph neural networks for blockchain transaction analysis
   - Ensemble methods with adaptive weighting

3. **Causal Modeling**
   - Causal inference methods for identifying cause-and-effect relationships
   - Structural equation models for understanding interrelationships
   - Interpretable neural networks with decision explanation

4. **Adaptive Learning**
   - Online learning with real-time model updates
   - Transfer learning between different cryptocurrencies and markets
   - Meta-learning for quick adaptation to new market conditions

5. **Improving Interpretability**
   - Methods for explaining predictions (SHAP, LIME)
   - Attention to certain parts of input data
   - Visualization of model internal workings

### Practical Recommendations

Based on the research conducted, several practical recommendations can be formulated:

1. **For Researchers**
   - Focus on multimodal approaches that combine various data sources
   - Develop methods for evaluating prediction uncertainty
   - Research methods to increase model interpretability

2. **For Traders**
   - Use GRU as one component of trading strategy, not the sole basis
   - Combine model signals with traditional analysis and risk management
   - Regularly retrain models when market regime changes

3. **For System Developers**
   - Implement data drift monitoring to detect model obsolescence
   - Develop easily updatable architectures with online learning capability
   - Include uncertainty evaluation mechanisms in trading systems

## Conclusion

GRU models demonstrate significant potential for cryptocurrency price forecasting due to their ability to capture complex temporal dependencies in data. Their simplified architecture compared to LSTM makes them more computationally efficient while maintaining comparable performance.

The key advantages of GRU in the context of cryptocurrency forecasting are:
- Ability to process long-term dependencies in time series
- Resilience to data noise
- Efficient training even on limited datasets
- Adaptability to different time scales

However, it's necessary to recognize the fundamental limitations of any forecasting method in the cryptocurrency market, including unpredictability of external events, high volatility, and data non-stationarity. The best results are achieved by combining GRU with other analysis methods, integrating heterogeneous data sources, and applying robust approaches to risk assessment and management.

Future research in this area should focus on increasing model interpretability, developing adaptive learning methods, and integration with causal modeling for better understanding the factors affecting cryptocurrency market dynamics.

## References

1. Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.

2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

3. Jing, W., Wang, T., Chen, C., & Xu, W. (2018). A gated recurrent unit approach to bitcoin price prediction. Journal of Risk and Financial Management, 11(4), 67.

4. McNally, S., Roche, J., & Caton, S. (2018). Predicting the price of Bitcoin using machine learning. In 2018 26th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP) (pp. 339-343). IEEE.

5. Alessandretti, L., ElBahrawy, A., Aiello, L. M., & Baronchelli, A. (2018). Anticipating cryptocurrency prices using machine learning. Complexity, 2018.

6. Siami-Namini, S., Tavakoli, N., & Namin, A. S. (2018). A comparison of ARIMA and LSTM in forecasting time series. In 2018 17th IEEE International Conference on Machine Learning and Applications (ICMLA) (pp. 1394-1401). IEEE.

7. Livieris, I. E., Pintelas, E., & Pintelas, P. (2020). A CNN–LSTM model for gold price time-series forecasting. Neural Computing and Applications, 32(23), 17351-17360.

8. Pang, X., Zhou, Y., Wang, P., Lin, W., & Chang, V. (2020). An innovative neural network approach for stock market prediction. The Journal of Supercomputing, 76(3), 2098-2118.

9. Mudassir, M., Bennbaia, S., Unal, D., & Hammoudeh, M. (2020). Time-series forecasting of Bitcoin prices using high-dimensional features: a machine learning approach. Neural Computing and Applications, 1-15.

10. Wu, C. H., Lu, C. C., Ma, Y. F., & Lu, R. S. (2018). A new forecasting framework for bitcoin price with LSTM. In 2018 IEEE International Conference on Data Mining Workshops (ICDMW) (pp. 168-175). IEEE.

11. Jaquart, P., Dann, D., & Weinhardt, C. (2021). Short-term bitcoin market prediction via machine learning. The Journal of Finance and Data Science, 7, 45-66.

12. Kristjanpoller, W., & Minutolo, M. C. (2018). A hybrid volatility forecasting framework integrating GARCH, artificial neural network, technical analysis and principal components analysis. Expert Systems with Applications, 109, 1-11.

13. Baek, Y., & Kim, H. Y. (2018). ModAugNet: A new forecasting framework for stock market index value with an overfitting prevention LSTM module and a prediction LSTM module. Expert Systems with Applications, 113, 457-480.

14. Li, J., Bu, H., & Wu, J. (2017). Sentiment-aware stock market prediction: A deep learning method. In 2017 International Conference on Service Systems and Service Management (pp. 1-6). IEEE.

15. Huang, J. Z., Huang, W., & Ni, J. (2019). Predicting bitcoin returns using high-dimensional technical indicators. The Journal of Finance and Data Science, 5(3), 140-155.
