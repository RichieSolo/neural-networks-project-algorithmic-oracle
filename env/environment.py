import gym
import numpy as np
from gym import spaces
import yfinance as yf
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import matplotlib.pyplot as plt

class PortfolioManagementEnv(gym.Env):
    def __init__(self, assets, initial_balance=1000000, window_size=60, risk_free_rate=0.01):
        super(PortfolioManagementEnv, self).__init__()
        self.assets = assets
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.action_dim = len(assets) + 1  # plus one for cash/bond holding
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.data = self.get_stock_data('2017-01-01')
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.window_size, len(assets) + 1), dtype=np.float32)  # Include cash in observation
        self.portfolio_returns = []
        self.risk_free_rate = risk_free_rate
        self.cash_holding = 0  # Additional state for holding cash

    def get_stock_data(self, start_date):
        data = {}
        for asset in self.assets:
            tickerData = yf.Ticker(asset)
            tickerDf = tickerData.history(period='1wk', start=start_date, end=pd.to_datetime(start_date) + pd.DateOffset(years=5))
            data[asset] = tickerDf['Close']
        return pd.DataFrame(data)

    def step(self, action):
        action = self.normalize_action(action)
        cash_percentage = action[-1]  # Last element of action array is cash holding percentage
        self.cash_holding = self.initial_balance * cash_percentage  # Update cash holding
        
        if self.current_step >= len(self.data) - 1:
            done = True
            reward = 0
            info = {"portfolio_value": self.initial_balance, "cash_holding": self.cash_holding}
            return np.zeros((self.window_size, len(self.assets) + 1)), reward, done, info

        window_data = self.data.iloc[self.current_step - self.window_size:self.current_step]
        windowed_returns = window_data.pct_change().fillna(0).values
        
        investable = self.initial_balance - self.cash_holding
        portfolio_return = np.dot(windowed_returns[-1], action[:-1]) * investable  # exclude cash in returns calculation
        self.portfolio_returns.append(portfolio_return)
        
        if len(self.portfolio_returns) > 1:
            mean_return = np.mean(self.portfolio_returns) - self.risk_free_rate
            std_dev = np.std(self.portfolio_returns)
            reward = mean_return / std_dev if std_dev != 0 else 0
        else:
            reward = 0  # Not enough data to calculate Sharpe Ratio

        self.initial_balance += portfolio_return  # Update total balance including returns from investments
        self.current_step += 1
        done = self.current_step >= len(self.data)
        info = {"portfolio_value": self.initial_balance, "cash_holding": self.cash_holding}
        next_state = np.hstack([windowed_returns, np.array([[self.cash_holding / self.initial_balance] * self.window_size]).T])
        return next_state, reward, done, info

    def reset(self):
        self.current_step = self.window_size
        self.initial_balance = 1000000
        self.portfolio_returns = []
        self.cash_holding = 0
        if self.current_step >= len(self.data):
            return np.zeros((self.window_size, len(self.assets) + 1))
        initial_returns = self.data.iloc[:self.window_size].pct_change().fillna(0).values
        initial_cash = np.array([[0] * self.window_size]).T  # Initialize cash holding state as zero
        return np.hstack([initial_returns, initial_cash])

    def normalize_action(self, action):
        total = np.sum(action)
        return action / total if total > 0 else np.zeros_like(action)


# Initialize environment and model as before
env = PortfolioManagementEnv(['AAPL', 'GOOGL', 'MSFT'])
model = PPO(MlpPolicy, env, verbose=1, batch_size=64, learning_rate=0.001, gae_lambda=0.95, gamma=0.99)

# Train the model
model.learn(total_timesteps=10000)

# Test the model
state = env.reset()
done = False
portfolio_values = []
cash_holdings = []
rewards = []
steps = []

step_count = 0
while not done:
    action, _ = model.predict(state, deterministic=True)
    next_state, reward, done, info = env.step(action)
    state = next_state
    # Append the relevant info for plotting
    portfolio_values.append(info['portfolio_value'])
    cash_holdings.append(info['cash_holding'])
    rewards.append(reward)
    steps.append(step_count)
    step_count += 1

# Creating plots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(steps, portfolio_values, label='Portfolio Value')
plt.xlabel('Steps')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(steps, cash_holdings, label='Cash Holdings')
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

