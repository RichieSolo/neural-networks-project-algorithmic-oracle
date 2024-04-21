import gym
import numpy as np
import pandas as pd
from gym import spaces
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

class PortfolioManagementEnv(gym.Env):
    """A custom environment for managing a stock portfolio in a reinforcement learning setup."""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, assets, start_date, initial_balance=1000000, lookback_weeks=12, risk_free_rate=0.0462):
        super().__init__()
        
        # list of stocks
        self.assets = assets
        
        # initial cash balance
        self.initial_balance = initial_balance
        
        # cash at time t 
        self.current_balance = initial_balance
        
        # how many weeks of data to look at
        self.lookback_weeks = lookback_weeks
        
        # risk free rate. Today's current risk free rate is 4.62%
        self.risk_free_rate = risk_free_rate
        
        # Action space is a 'real' number in (0,1)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(assets) + 1,), dtype=np.float32)
        
        # Download stock data
        self.stock_data = self.download_stock_data(start_date)
        
        # Observation space is a vector of stock prices, volumes, and risks for the past lookback_weeks weeks
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.lookback_weeks * len(assets) * 3,), dtype=np.float32)
        
        # how much cash on 'hand'
        self.cash_held = 0
        
        # current step
        self.current_step = 60

    def download_stock_data(self, start_date):
        # Download historical data for each asset and concatenate into a single DataFrame
        data_frames = []
        for asset in self.assets:
            ticker = yf.Ticker(asset)
            historical_data = ticker.history(period='1d', start=start_date, end=pd.to_datetime(start_date) + pd.DateOffset(years=5))
            historical_data['Ticker'] = asset
            data_frames.append(historical_data[['Close', 'Volume']])
        full_data = pd.concat(data_frames, keys=self.assets, names=['Ticker', 'Date'])
        full_data = full_data.unstack(level=0).swaplevel(i=0, j=1, axis=1).sort_index(axis=1)
        return full_data

    def calculate_weekly_stats(self):
        # Calculate weekly statistics for each asset
        weekly_data = {}
        for asset in self.assets:
            asset_data = self.stock_data[asset]
            close_price_60d = asset_data['Close'].rolling(window=self.lookback_weeks).mean()
            volume_60d = asset_data['Volume'].rolling(window=self.lookback_weeks).mean()
            risk_60d = asset_data['Close'].rolling(window=self.lookback_weeks).std()
            weekly_data[asset] = pd.DataFrame({'Close': close_price_60d, 'Volume': volume_60d, 'Risk': risk_60d})
        combined_weekly_data = pd.concat(weekly_data, axis=1).dropna()
        return combined_weekly_data

    def reset(self, seed=None):
        # Reset the environment to its initial state
        if seed is not None:
            np.random.seed(seed)

        self.current_balance = self.initial_balance
        self.cash_held = 0
        self.current_step = self.lookback_weeks  # Set this to a valid index that ensures data availability

        weekly_stats = self.calculate_weekly_stats()
        if self.current_step < len(weekly_stats):
            initial_state = weekly_stats.iloc[self.current_step - self.lookback_weeks:self.current_step].values.flatten().astype(np.float32)
        else:
            # Fallback to zeros if the index is out of bounds
            initial_state = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        info = {}  # Add an empty info dictionary
        return initial_state, info  # Return a tuple of observation and info

    def step(self, action):
        # Execute one time step within the environment
        normalized_action = self.normalize_action(action)
        self.cash_held = self.current_balance * normalized_action[-1]
        
        weekly_stats = self.calculate_weekly_stats()
        investment_indices = np.s_[0:len(self.assets)]
        investment_metrics = weekly_stats.iloc[self.current_step, investment_indices].values.astype(np.float32).flatten()
        
        # Normalize investment metrics
        investment_metrics = (investment_metrics - np.mean(investment_metrics)) / (np.std(investment_metrics) + 1e-8)
        
        # Allocate the remaining balance to stocks based on the normalized action
        stock_allocation = self.current_balance * normalized_action[:-1]
        investment_return = np.dot(investment_metrics, stock_allocation)
        self.current_balance = self.cash_held + np.sum(stock_allocation * (1 + investment_metrics))
        
        reward = float(investment_return / (np.std(investment_metrics) + 1e-8))
        
        self.current_step += 1
        done = self.current_step >= len(weekly_stats)
        terminated = done
        truncated = False
        
        info = {"current_balance": self.current_balance, "cash_held": self.cash_held}
        
        # Check if there is enough data for the next state
        if self.current_step + self.lookback_weeks <= len(weekly_stats):
            next_state = weekly_stats.iloc[self.current_step:self.current_step + self.lookback_weeks].values.astype(np.float32).flatten()
        else:
            # Not enough data, pad with zeros or repeat last available data
            last_available_data = weekly_stats.iloc[-self.lookback_weeks:].values.astype(np.float32).flatten()
            next_state = np.pad(last_available_data, (0, max(0, self.lookback_weeks * len(self.assets) * 3 - last_available_data.size)), 'constant', constant_values=0)
        
        # Ensure next_state has the correct shape
        next_state = next_state.reshape((self.lookback_weeks * len(self.assets) * 3,))
        
        return next_state, reward, terminated, truncated, info

    def normalize_action(self, action):
        # Normalize the action using the softmax function
        action = np.exp(action - np.max(action))  # Apply softmax function
        return action / np.sum(action)

# Environment initialization and training
assets = ['AAPL', 'GOOGL', 'MSFT']
start_date = '2017-01-01'
end_date = pd.to_datetime(start_date) + pd.DateOffset(years=5)
train_end_date = pd.to_datetime(start_date) + pd.DateOffset(years=4)

# Download historical data for each asset and concatenate into a single DataFrame
data_frames = []
for asset in assets:
    ticker = yf.Ticker(asset)
    historical_data = ticker.history(period='1d', start=start_date, end=end_date)
    historical_data['Ticker'] = asset
    data_frames.append(historical_data[['Close', 'Volume']])

full_data = pd.concat(data_frames, keys=assets, names=['Ticker', 'Date'])
full_data = full_data.unstack(level=0).swaplevel(i=0, j=1, axis=1).sort_index(axis=1)

# Localize the datetime objects to the same timezone as the data
start_date = pd.to_datetime(start_date).tz_localize(full_data.index.tz)
end_date = end_date.tz_localize(full_data.index.tz)
train_end_date = train_end_date.tz_localize(full_data.index.tz)

# Split data into training and testing periods
train_data = full_data[start_date:train_end_date]
test_data = full_data[train_end_date:end_date]

# Environment initialization and training
env = PortfolioManagementEnv(assets, start_date)
env.stock_data = train_data  # Use training data for the environment
check_env(env)

vec_env = DummyVecEnv([lambda: env])
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=20000)
model.save("ppo_portfolio_management")

# Initialize variables for tracking
steps = []
portfolio_values = []
amount_holdings = []
rewards = []

# Simulate the environment on the testing data
env.stock_data = test_data  # Use testing data for the environment
env.current_step = env.lookback_weeks  # Reset the current step
obs, _ = env.reset()  # Get the initial observation and ignore the info

for step in range(len(test_data) - env.lookback_weeks):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)  # Use env.step() instead of vec_env.step()
    steps.append(step)
    portfolio_values.append(info['current_balance'])
    amount_holdings.append(info['cash_held'])
    rewards.append(reward)
    if done:
        break

# Creating plots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(steps, portfolio_values, label='Portfolio Value')
plt.xlabel('Steps')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(steps, amount_holdings, label='Cash Holdings')
plt.xlabel('Steps')
plt.ylabel('Cash Holdings')
plt.title('Cash Holdings Over Time')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(steps, rewards, label='Rewards')
plt.xlabel('Steps')
plt.ylabel('Rewards')
plt.title('Rewards Over Time')
plt.grid(True)

plt.tight_layout()
plt.show()