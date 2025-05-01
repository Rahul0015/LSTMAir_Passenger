import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Month'] = pd.to_datetime(df['Month'])
    df['Year'] = df['Month'].dt.year
    df['Month_Num'] = df['Month'].dt.month
    return df

def decompose_series(df):
    result = seasonal_decompose(df['#Passengers'], model='multiplicative', period=12)
    result.plot()
    plt.tight_layout()
    plt.show()

def prepare_sequences(df, time_steps=10):
    scaler = MinMaxScaler()
    df['#Passengers'] = scaler.fit_transform(df[['#Passengers']])
    data = df['#Passengers'].values

    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])

    X = np.array(X).reshape(-1, time_steps, 1)
    y = np.array(y)

    return X, y, scaler
