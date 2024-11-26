import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def main():
    # Example data: User's monthly expenses
    data = [2000, 2200, 2100, 2300, 2500, 2400, 2600, 2500, 2700, 2800]
    data = np.array(data).reshape(-1, 1)
    # Prepare data
    seq_len = 3
    X = np.array([data[i:i + seq_len].flatten() for i in range(len(data) - seq_len)])
    y = data[seq_len:]

    # Reshape for LSTM
    X = X.reshape((X.shape[0], seq_len, 1))

    # Define LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_len, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    # Train model
    model.fit(X, y, epochs=100, batch_size=1)
    # Predict future spending
    future_input = np.array([2600, 2500, 2700]).reshape(1, seq_len, 1)
    predicted_spending = model.predict(future_input)
    print(predicted_spending)

if __name__ == "__main__":
    main()