import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

start_date = st.date_input('Start Date', value=pd.to_datetime('2010-01-01'))
end_date = st.date_input('End Date', value=pd.to_datetime('2019-12-31'))


# Button to trigger the data fetching and processing
if st.button('Predict Stock Trend'):
    # Fetching data using yfinance
    df = yf.download(user_input, start=start_date, end=end_date)

    # Check if the DataFrame is empty
    if df.empty:
        st.error('Failed to fetch data. Please check the stock ticker and date range.')
    else:
        # Describing Data
        st.subheader(f'Data from {start_date} to {end_date}')
        st.write(df.describe())

    # Visualization
    st.subheader('Closing Price vs. Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, label='Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Closing Price vs. Time Chart with 100 Moving Average')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, label='Closing Price')
    plt.plot(ma100, label='100-day MA', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Closing Price vs. Time Chart with 100 & 200 Moving Average')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, label='Closing Price')
    plt.plot(ma100, label='100-day MA', color='orange')
    plt.plot(ma200, label='200-day MA', color='magenta')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    st.pyplot(fig)

    # Splitting Data into Training & Testing
    training_data = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    testing_data = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

    # Scaling the data for further analysis
    scaler = MinMaxScaler(feature_range=(0, 1))
    training_data_array = scaler.fit_transform(training_data)

    # Load the model
    model = load_model('keras.h5')

    # Testing part
    past_100_days = training_data.tail(100)
    final_df = pd.concat([past_100_days, testing_data], ignore_index=True)
    input_data = scaler.transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    y_predicted = model.predict(x_test)
    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Final Graph
    st.subheader('Predicted vs Original')

    fig2 = plt.figure(figsize=(14, 8))
    plt.plot(y_test, color='blue', linestyle='-', linewidth=2.5, label='Original Price')
    plt.plot(y_predicted, color='red', linestyle='--', linewidth=2.5, label='Predicted Price')

    plt.xlabel('Time', fontsize=14, fontweight='bold', fontname='Arial')
    plt.ylabel('Price', fontsize=14, fontweight='bold', fontname='Arial')

    plt.title('Predicted vs Original Stock Prices', fontsize=16, fontweight='bold', fontname='Arial', color='darkgreen')

    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12, colors='black')

    st.pyplot(fig2)
