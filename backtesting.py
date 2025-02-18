import requests
import os
import pandas as pd
import numpy as np
import ta
import datetime as dt
from datetime import datetime, timedelta
import pytz

# Load access token from environment variable
access_token = os.getenv("ACCESS_TOKEN")
angle_threshold = 15.0
expiry_date = "2025-02-20"
from_date = "2025-02-13"
to_date = "2025-02-16"

class UpstoxBacktest:
    def __init__(self, access_token):
        self.access_token = access_token
        self.expiry_date = expiry_date
        self.from_date = from_date
        self.to_date = to_date
        self.angle_threshold = angle_threshold
        self.trades = []
        self.instrument_df = self.load_instruments()

    def load_instruments(self):
        df = pd.read_csv("https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz")
        df = df[(df['name'] == "NIFTY") & (df['instrument_type'] == "OPTIDX")]
        df['expiry'] = pd.to_datetime(df['expiry'])
        df.reset_index(drop=True, inplace=True)
        return df

    def get_historical_data(self, instrument_key, from_date=None, to_date=None):
        if not from_date:
            from_date = (dt.datetime.now() - dt.timedelta(days=4)).date()
        if not to_date:
            to_date = dt.datetime.now().date()

        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/1minute/{to_date}/{from_date}"
        headers = {'Accept': 'application/json', 'Authorization': f'Bearer {self.access_token}'}
        response = requests.get(url, headers=headers)
        res = response.json()

        if 'data' not in res or 'candles' not in res['data']:
            print(f"Error fetching data for {instrument_key}: {res.get('errors', 'Unknown error')}")
            return pd.DataFrame()

        candle_data = res['data']['candles']
        df = pd.DataFrame(candle_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata')
        df.sort_values('timestamp', inplace=True)

        return df[['timestamp', 'open', 'high', 'low', 'close']]

    def calculate_indicators(self, df):
        df['EMA9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['EMA15'] = ta.trend.ema_indicator(df['close'], window=15)
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['MACD'] = ta.trend.macd(df['close'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['close'])
        df['EMA_Angle'] = self.calculate_ema_angle(df)

        # Calculate candle strength
        df['candle_body'] = abs(df['close'] - df['open'])
        high_low_diff = df['high'] - df['low']
        df['candle_strength'] = np.where(high_low_diff > 0, df['candle_body'] / high_low_diff, 0)

        return df

    def calculate_ema_angle(self, df):
        ema9_slope = df['EMA9'].diff()
        ema15_slope = df['EMA15'].diff()
        ema_angle = np.arctan(np.abs(ema9_slope / ema15_slope))
        return pd.Series(np.degrees(ema_angle), index=df.index)  # Ensure proper alignment

    def get_nearest_option(self, nifty_price, expiry_date=None):
        nearest_strike = round(nifty_price / 50) * 50

        if expiry_date:
            expiry_date = pd.to_datetime(expiry_date)
            expiry_filtered_df = self.instrument_df[self.instrument_df['expiry'] == expiry_date]
        else:
            expiry_filtered_df = self.instrument_df[self.instrument_df['expiry'] >= datetime.now()]

        if expiry_filtered_df.empty:
            print(f"No options available for expiry {expiry_date if expiry_date else 'nearest expiry'}")
            return None, None

        nearest_expiry = expiry_filtered_df['expiry'].min()
        expiry_filtered_df = expiry_filtered_df[expiry_filtered_df['expiry'] == nearest_expiry]

        ce_option = expiry_filtered_df[(expiry_filtered_df['strike'] == nearest_strike) & (expiry_filtered_df['option_type'] == 'CE')]
        pe_option = expiry_filtered_df[(expiry_filtered_df['strike'] == nearest_strike) & (expiry_filtered_df['option_type'] == 'PE')]

        if ce_option.empty or pe_option.empty:
            return None, None

        return ce_option.iloc[0]['instrument_key'], pe_option.iloc[0]['instrument_key']

    def test_conditions(self, df, index):
        row = df.loc[index]
        prev_row = df.shift(1).loc[index]

        # Check EMA crossover (EMA9 crossing above EMA15 for bullish, below for bearish)
        ema_crossover = None
        if prev_row['EMA9'] < prev_row['EMA15'] and row['EMA9'] > row['EMA15']:
            ema_crossover = 'bullish'
        elif prev_row['EMA9'] > prev_row['EMA15'] and row['EMA9'] < row['EMA15']:
            ema_crossover = 'bearish'

        # If no EMA crossover, return None
        if ema_crossover is None:
            return None

        # Check EMA angle condition (should be greater than or equal to 15 degrees)
        if abs(row['EMA_Angle']) < angle_threshold:
            return None  

        # **New Dynamic Price Retest Condition (Price within ±10 to ±15 points of EMA9)**
        lower_bound = row['EMA9'] - 10
        upper_bound = row['EMA9'] + 10
        if not (lower_bound <= row['close'] <= upper_bound):
            return None

        # Check candle strength condition
        if row['candle_strength'] <= 0.6:
            return None

        # Check for Bullish Signal
        if ema_crossover == 'bullish' and row['MACD'] > row['MACD_Signal'] and row['EMA9'] > row['EMA15'] and 30 <= row['RSI'] <= 70 and row['ADX'] > 20:
            return "BULLISH"

        # Check for Bearish Signal
        elif ema_crossover == 'bearish' and row['MACD'] < row['MACD_Signal'] and row['EMA9'] < row['EMA15'] and 30 <= row['RSI'] <= 70 and row['ADX'] > 20:
            return "BEARISH"

        return None

    def run_backtest(self, from_date, to_date, expiry_date=None, angle_threshold=None):
        df_nifty = self.get_historical_data("NSE_INDEX|Nifty 50", from_date, to_date)
        if df_nifty.empty:
            return pd.DataFrame()

        df_nifty.set_index('timestamp', inplace=True)
        df_nifty = self.calculate_indicators(df_nifty)

        position = None
        quantity = 75

        for index, row in df_nifty.iterrows():
            signal = self.test_conditions(df_nifty, index)

            # Check for open positions before entering a new trade
            if position is None and signal: 
                ce_key, pe_key = self.get_nearest_option(row['close'], expiry_date)
                option_key = ce_key if signal == "BULLISH" else pe_key

                if option_key:
                    df_option = self.get_historical_data(option_key, from_date, to_date)
                    if not df_option.empty:
                        df_option.set_index('timestamp', inplace=True)
                        if index in df_option.index:
                            entry_price = df_option.loc[index, 'close']

                            option_details = self.instrument_df[self.instrument_df['instrument_key'] == option_key]
                            if not option_details.empty:
                                trading_symbol = option_details.iloc[0]['tradingsymbol']

                            position = {
                                'entry_price': entry_price,
                                'entry_time': index,
                                'stop_loss': entry_price - 10,
                                'target': entry_price + 30,
                                'instrument_key': option_key,
                                'tradingsymbol': trading_symbol  # Add trading symbol to position dictionary
                            }

                            self.trades.append({
                                'entry_time': position['entry_time'],
                                'entry_price': position['entry_price'],
                                'exit_time': None,
                                'exit_price': None,
                                'pnl': None,
                                'reason': 'Trade Entered',
                                'tradingsymbol': position['tradingsymbol']  # Add trading symbol to trades list
                            })

            # Check if the previous trade is closed before entering a new trade
            elif position and (row['low'] <= position['stop_loss'] or row['high'] >= position['target']):
                df_option = self.get_historical_data(position['instrument_key'], from_date, to_date)
                if not df_option.empty:
                    df_option.set_index('timestamp', inplace=True)

                    for i, option_row in df_option.loc[index:].iterrows():
                        if option_row['low'] <= position['stop_loss']:
                            exit_price = position['stop_loss']
                            reason = "Stop Loss Hit"
                            break
                        elif option_row['high'] >= position['target']:
                            exit_price = position['target']
                            reason = "Target Hit"
                            break
                    else:
                        continue  

                    pnl = (exit_price - position['entry_price']) * quantity

                    self.trades[-1]['exit_time'] = i
                    self.trades[-1]['exit_price'] = exit_price
                    self.trades[-1]['pnl'] = pnl
                    self.trades[-1]['reason'] = reason

                    position = None

        results = pd.DataFrame(self.trades)
        return results
backtest = UpstoxBacktest(access_token)
results = backtest.run_backtest(from_date, to_date, expiry_date, angle_threshold)
print("\nBacktest Results:")
print(f"Overall PnL: {results['pnl'].sum() if not results.empty else 0}")
if not results.empty:
    print(results.to_string())

    print("\nBacktest Completed🚀'")
else:
    print("No trades executed.")