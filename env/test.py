
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
import numpy as np
from my_utils.data import *
from gymnasium import spaces
import math

df = get_sp500_df()

print(df.columns)

def get_stock_data(df, stock_symbol, start_date):
    # Get the data for the stock
    stock_data = df[df['Symbol'] == stock_symbol]
    
    # Get the 60 days of previous data
    prev_data = stock_data[stock_data['Date'] < start_date].tail(60)
    
    # Get the 10 years of data starting from the start date
    data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= (pd.to_datetime(start_date) + pd.DateOffset(years=10)))]
    
    # Concatenate the previous data and the 10 years of data
    data = pd.concat([prev_data, data])
    
    # Increment the start date by one week
    next_start_date = pd.to_datetime(start_date) + pd.DateOffset(weeks=1)
    
    return data, next_start_date

start_date = '2010-01-04'
stock_symbol = 'AAPL'

while True:
    data, next_start_date = get_stock_data(df, stock_symbol, start_date)
    # Do something with the data
    print(data)
    # Increment the start date
    start_date = next_start_date.strftime('%Y-%m-%d')
    # Stop after 10 years
    if pd.to_datetime(start_date) >= pd.to_datetime('2020-01-04'):
        break