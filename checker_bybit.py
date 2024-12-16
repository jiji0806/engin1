########################################################################################################
# Market Starter!
########################################################################################################
import os.path
import ccxt
import traceback
import numpy as np
from dateutil import parser
from scipy.signal import find_peaks
import json
import time, sys
import math
import datetime as dt
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import pytz
#from cs50 import SQL
import sqlite3
import boto3
from botocore.config import Config
import datetime
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
import operator
import subprocess, psutil, re
import traceback
from finta import TA
import itertools
from findpeaks import findpeaks
from datetime import timedelta
import requests, base64

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model  # 수정된 부분: 모델 불러오기
from adtk.detector import MinClusterDetector
from sklearn.cluster import KMeans
from adtk.data import validate_series
from adtk.detector import OutlierDetector
from sklearn.neighbors import LocalOutlierFactor
from adtk.detector import RegressionAD
from sklearn.linear_model import LinearRegression
from adtk.detector import PcaAD
from adtk.detector import QuantileAD

# AnomalyDetection
from scipy import sparse, stats
from sklearn.metrics import f1_score

dynamodb_table_name = 'worldengin_configuration'
####################################################################################################################################
trader_name = 'binanceusdm_perpetual_1' # delivatives_futures_usdt_perpetual
####################################################################################################################################
#boto3 ddb
my_config = Config(
    region_name = 'ap-northeast-1',
    signature_version = 'v4',
    retries = {
        'max_attempts': 10,
        'mode': 'standard'
    }
)

def get_configuration(trader_name, dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb', config=my_config)
    table = dynamodb.Table(dynamodb_table_name)
    try:
        response = table.get_item(Key={'trader_name': trader_name})
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        return response['Item']

def send_to_telegram(*messages):
    message = "\n".join(messages)
    telegram_bot_token = '6521282593:AAEJWTaV6qavLac7Xu9_-iG_neHxm53F8KM'
    telegram_chat_id = '1051301724'
    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage?chat_id={telegram_chat_id}&text={message}"
    response = requests.get(url)

def ps_counter(name):
    p_id_list = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'cmdline']):
        if proc.info["cmdline"] is not None and (name in proc.info["cmdline"]) and (proc.info["name"] == "python3"):
            print(f'-현재 러닝 프로세서 감지: {proc.info["cmdline"]}')
            p_id_list.append(proc.info["pid"])
    return len(p_id_list), p_id_list

def position_calc(exchange_id, symbol_):
    position_size = 0
    position_side = ''
    position_value = 0
    position_entry_price = ''
    liquidation_price = ''
    unrealised_pnl = 0
    roe_pcnt = 0
    if (exchange_id == 'binanceusdm') or (exchange_id == 'bybit') or (exchange_id == 'ftx') or (exchange_id == 'bitmex'):
        position_checker_ = exchange.fetch_positions([symbol_])
        if position_checker_:
            position_checker = position_checker_[0]
            position_size = position_checker["contracts"] # always positive number
            if position_size != 0:
                position_side = position_checker["side"]
                position_value = float(position_checker["notional"])
                position_entry_price = position_checker["entryPrice"]
                liquidation_price = position_checker["liquidationPrice"]
                unrealised_pnl = position_checker["unrealizedPnl"]
                roe_pcnt = position_checker["percentage"]
                if (position_side == 'short') and position_value > 0:
                    position_value = position_value * -1
                if exchange_id == 'bitmex':
                    if (position_size == 0) or (position_size == None):
                        position_size = 0
                        position_side = ''
                        position_value = 0
                        position_entry_price = ''
                        liquidation_price = ''
                    if position_size > 0:
                        position_side = 'long'
                    elif position_size < 0:
                        position_size = position_size * -1
                        position_side = 'short'


    #elif exchange_id == 'bybit':
    #    position_checker = exchange.fetch_positions([symbol_])[0]
    #    position_size = float(position_checker["size"])
    #    if position_size != 0:
    #        if position_checker["side"] == 'Buy':
    #            position_side = 'long'
    #        if position_checker["side"] == 'Sell':
    #            position_side = 'short'        
    #        position_value = float(position_checker["position_value"])
    #        position_entry_price = float(position_checker["entry_price"])
    #        liquidation_price = float(position_checker["liq_price"])
    #        unrealised_pnl = float(position_checker["unrealised_pnl"])
    #        roe_pcnt = unrealised_pnl / (position_entry_price / float(position_checker["leverage"]) * position_size) * 100

    elif (exchange_id == 'huobi'):
        if len(exchange.fetch_positions([symbol_])) > 0:
            position_checker = exchange.fetch_positions([symbol_])[0]
            position_size = float(position_checker["contractSize"] * position_checker["contracts"])
            if position_size != 0:
                position_side = position_checker["side"]
                position_value = position_checker["notional"]
                position_entry_price = position_checker["entryPrice"]
                liquidation_price = position_checker["liquidationPrice"] # none
                unrealised_pnl = position_checker["unrealizedProfit"]
                roe_pcnt = position_checker["percentage"]
        else:
            position_size = 0
            position_side = ''
            position_value = 0
            position_entry_price = ''
            liquidation_price = ''
            unrealised_pnl = 0
            roe_pcnt = 0

    #elif exchange_id == 'bitmex':
    #    position_checker = exchange.fetch_positions([symbol_])[0]
    #    position_size = float(position_checker['info']['currentQty'])
    #    if position_size != 0:
    #        if position_size > 0:
    #            position_side = 'long'
    #            position_value = float(position_checker['info']['homeNotional'])
    #        elif position_size < 0:
    #            position_size = position_size*-1
    #            position_side = 'short'
    #            position_value = float(position_checker['info']['homeNotional'])*-1
    #        position_entry_price = float(position_checker['info']['avgEntryPrice'])
    #        liquidation_price= float(position_checker['info']['liquidationPrice'])
    #        unrealised_pnl= float(position_checker['info']['unrealisedPnl'])/100000000
    #        roe_pcnt= float(position_checker['info']['unrealisedRoePcnt'])*100

    if position_value is None:
        position_value = 0
    elif (isinstance(position_value, int)) or (isinstance(position_value, float)):
        position_value = position_value
    if position_entry_price is None:
        position_entry_price = '-'
    elif (isinstance(position_entry_price, int)) or (isinstance(position_entry_price, float)):
        position_entry_price = position_entry_price
    if liquidation_price is None:
        liquidation_price = '-'
    elif (isinstance(liquidation_price, int)) or (isinstance(liquidation_price, float)):
        liquidation_price = liquidation_price
    if unrealised_pnl is None:
        unrealised_pnl = 0
    elif (isinstance(unrealised_pnl, int)) or (isinstance(unrealised_pnl, float)):
        unrealised_pnl = unrealised_pnl
    if roe_pcnt is None:
        roe_pcnt = 0
    elif (isinstance(roe_pcnt, int)) or (isinstance(roe_pcnt, float)):
        roe_pcnt = roe_pcnt

    return position_side, position_size, position_value, position_entry_price, liquidation_price, unrealised_pnl, roe_pcnt

def klines(symbol, interval, limit):
    klines = exchange.fetch_ohlcv(symbol=symbol, timeframe=interval, limit=limit)
    close_time_of_the_last_row = datetime.datetime.now()
    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
    df['close_time'] = df['open_time'].shift(-1)
    df.at[df.index[-1], 'close_time'] = close_time_of_the_last_row  # Set the close_time of the last row to current time
    df['open_time2'] = df['open_time']
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
    df['open_time2'] = pd.to_datetime(df['open_time2'], unit='ms', errors='coerce')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', errors='coerce')

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce').astype('datetime64[ms]')
    df['open_time2'] = pd.to_datetime(df['open_time2'], unit='ms', errors='coerce').astype('datetime64[ms]')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', errors='coerce').astype('datetime64[ms]')

    df['open_time'] = df['open_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    df['open_time2'] = df['open_time2'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    df['close_time'] = df['close_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    
    df['peak_time'] = df['open_time']

    df['close_high'] = df['close']
    df.at[df.index[-2], 'close_high'] = df['high'].iloc[-1]
    df['close_low'] = df['close']
    df.at[df.index[-2], 'close_low'] = df['low'].iloc[-1]

    df = df.set_index('open_time')
    return df

def peak_calc(symbol, intervals, limit):
    for interval in intervals:
        df=klines(symbol=symbol, interval=interval, limit=limit)
        if(len(df) > 600):
            CustomStrategy = ta.Strategy( # Create your own Custom Strategy
                name="Momo and Volatility",
                description="RSI, MACD, adx, dmp, dmn, obv, atr",
                ta=[
                    {"kind": "adx"},
                    {"kind": "adx", "length": 200},
                    {"kind": "bbands", "length": 21},
                    {"kind": "atr"},
                    {"kind": "atr", "length": 1},
                ]
            )

            df.ta.strategy(CustomStrategy) # To run your "Custom Strategy"
            df['atr_diff'] = df['ATRr_14'].diff()
            df['atr_p'] = df['ATRr_14']/df['close']*100
            df['atr_p_diff'] = df['atr_p'].diff()
            df['adx_diff'] = df['ADX_14'].diff()
            df['dx_diff'] = df['DX_14'].diff()










                ########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
                ###################################################################################################
                ###################################################################################################



            iqr_close = None
            iqr_feature1 = None
            iqr_feature2 = None

            df['feature1'] = 0
            df['feature2'] = 0

            df['anomalies_close'] = False
            df['anomalies_feature1'] = False
            df['anomalies_feature2'] = False

            df['maxima_peak_x_anomalies_close'] = 0
            df['minima_peak_x_anomalies_close'] = 0
            df['maxima_peak_x_anomalies_feature1'] = 0
            df['minima_peak_x_anomalies_feature1'] = 0
            df['maxima_peak_x_anomalies_feature2'] = 0
            df['minima_peak_x_anomalies_feature2'] = 0



            df_cleaned = df[[ \

                'BBB_21_2.0', \
                'volume', \
                'ATRr_1', \
                





                
            ]].select_dtypes(include=np.number).dropna()


            # # 데이터 준비
            # columns = df_cleaned.columns
            # X = df_cleaned[columns].values

            # 데이터 준비
            columns = df_cleaned.columns
            # X = df_cleaned[columns].values
            desired_columns = ['BBB_21_2.0', 'volume', 'ATRr_1']
            # desired_columns = ['ATRr_1', 'ATRr_14']
            X = df_cleaned[desired_columns].values
            




            # 데이터 전처리
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(X)

            # 저장된 모델 불러오기
            # print(interval)
            loaded_autoencoder = load_model("model_1.h5")
            
            # loaded_autoencoder.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
            # Compile the model (assuming you have defined loss and optimizer during training)


            # 학습된 오토인코더를 사용하여 데이터를 변환
            # 수정된 부분: scaled_data 대신에 새로운 데이터를 사용하여 변환
            encoded_data_mlp = loaded_autoencoder.predict(scaled_data)  # 수정된 부분: 저장된 모델을 사용하여 데이터 변환

            # PCA를 사용하여 데이터를 2차원으로 축소
            pca = PCA(n_components=2)
            encoded_data_2d = pca.fit_transform(encoded_data_mlp)

            df.loc[df_cleaned.index, 'feature1'] = encoded_data_2d[:, 0]  # df_cleaned의 인덱스를 기준으로 값 할당
            df.loc[df_cleaned.index, 'feature2'] = encoded_data_2d[:, 1]

            df['feature1_diff'] = df['feature1'].diff()
            df['feature2_diff'] = df['feature2'].diff()
            df['feature1_percentage_change'] = np.where(df['feature1_diff'].abs().shift() != 0, (df['feature1_diff'].diff() / df['feature1_diff'].abs().shift()) * 100, 0)
            df['feature2_percentage_change'] = np.where(df['feature2_diff'].abs().shift() != 0, (df['feature2_diff'].diff() / df['feature2_diff'].abs().shift()) * 100, 0)

                ########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
                ###################################################################################################
                ###################################################################################################




















            globals()['df_'+interval] = df

    return




def get_available_slots(base_markets, running_markets, new_ranked_markets, out_ranked_markets):
    available_slots = base_markets.copy()
    
    # Remove slots occupied by running markets
    for market_slot, _ in running_markets.items():
        if market_slot in available_slots:
            del available_slots[market_slot]
    
    # Add back slots available due to out_ranked_markets
    for market_slot, _ in out_ranked_markets.items():
        available_slots[market_slot] = ''
    
    return available_slots


def get_markets_to_start(available_slots, new_ranked_markets):
    markets_to_start = {}
    slots_to_fill = list(available_slots.keys())
    
    # Fill available slots with new ranked markets
    for slot, market in zip(slots_to_fill, new_ranked_markets.values()):
        markets_to_start[slot] = market
    
    return markets_to_start

def balance_calc(exchange_id, balance_currency):
    if (exchange_id == 'binanceusdm') or (exchange_id == 'ftx'):
        # wallet_balance = exchange.fetch_balance()[balance_currency]['total'] #binance, bitmex
        wallet_balance = float(exchange.fetch_balance()['info']['totalMarginBalance'])
    elif exchange_id == 'bybit':
        wallet_balance = exchange.fetch_balance()['total'][balance_currency] #bybit
    elif exchange_id == 'bitmex':
        wallet_balance = exchange.fetch_balance()[balance_currency]['total'] #binance, bitmex
    elif exchange_id == 'huobi':
        wallet_balance = float(exchange.fetch_balance()['info']['data'][0]['margin_balance']) # huobi
    return wallet_balance


def ticker_calc(symbol_):
    if (exchange_id == 'ftx'):
        symbol_ticker = exchange.fetch_ticker(symbol_)
        symbol_ticker_timestamp = 0
        symbol_ticker_last = symbol_ticker['last']
        last_24hr_volatility = 0
    else:
        symbol_ticker = exchange.fetch_ticker(symbol_)
        symbol_ticker_timestamp = symbol_ticker['timestamp']
        symbol_ticker_last = symbol_ticker['last']
        last_24hr_volatility = symbol_ticker['change']
    return symbol_ticker_timestamp, symbol_ticker_last, last_24hr_volatility


def l_p_l_and_l_p_h_calc(symbol_ticker_last):
    # if (globals()['atr_pick'] != ''):
    #     interval_ = globals()['atr_pick']
    # else:
    #     interval_ = '15m'

    interval_ = '5m'
    l_p_l = globals()['df_' + interval_]['low'].tail(2).min() # 15m기준 마지막 3개 중 가장 낮은가
    l_p_h = globals()['df_' + interval_]['high'].tail(2).max() # 15m기준 마지막 3개 중 가장 높은가
    last_peaked_price = symbol_ticker_last
    return l_p_l, l_p_h, last_peaked_price




def interval_cumulate_lv(df, cumulate_lv):
    if 'SMA_7' in df.columns:
        cumulate_lv.append(df.SMA_7.iloc[-1])
    if 'SMA_30' in df.columns:
        cumulate_lv.append(df.SMA_30.iloc[-1])
    if 'SMA_50' in df.columns:
        cumulate_lv.append(df.SMA_50.iloc[-1])
    # if 'SMA_100' in df.columns:
    #     cumulate_lv.append(df.SMA_100.iloc[-1])
    if 'SMA_200' in df.columns:
        cumulate_lv.append(df.SMA_200.iloc[-1])
    if 'EMA_7' in df.columns:
        cumulate_lv.append(df.EMA_7.iloc[-1])
    if 'EMA_30' in df.columns:        
        cumulate_lv.append(df.EMA_30.iloc[-1])
    if 'EMA_50' in df.columns:
        cumulate_lv.append(df.EMA_50.iloc[-1])
    # if 'EMA_100' in df.columns:
    #     cumulate_lv.append(df.EMA_100.iloc[-1])
    if 'EMA_200' in df.columns:
        cumulate_lv.append(df.EMA_200.iloc[-1])
    if 'BBU_21_2.0' in df.columns:
        cumulate_lv.append(df['BBU_21_2.0'].iloc[-1])
    if 'BBL_21_2.0' in df.columns:
        cumulate_lv.append(df['BBL_21_2.0'].iloc[-1])
    if 'PSARl_0.02_0.2' in df.columns:
        cumulate_lv.append(df['PSARl_0.02_0.2'].iloc[-1])
    if 'PSARs_0.02_0.2' in df.columns:
        cumulate_lv.append(df['PSARs_0.02_0.2'].iloc[-1])
        
    cumulate_lv.append(df['high'].max() * (1 - 0.004))
    cumulate_lv.append(df['low'].min() * (1 + 0.004))

    return cumulate_lv

def cumulate_lv_calc(symbol_, intervals):
    cumulate_lv =[]
    print(symbol_)
    print(intervals)
    for interval in intervals:
        print(interval)
        df=klines(symbol=symbol_, interval=interval, limit=limit)
        CustomStrategy = ta.Strategy( # Create your own Custom Strategy
            name="Momo and Volatility",
            description="SMA 22, 50, 100, 200, 999 and EMA 22, 50, 100, 200, 999 BBANDS, RSI, MACD",
            ta=[
                {"kind": "sma", "length": 7},
                {"kind": "sma", "length": 30},
                {"kind": "sma", "length": 50},
                # {"kind": "sma", "length": 100},
                {"kind": "sma", "length": 200},
                {"kind": "ema", "length": 7},
                {"kind": "ema", "length": 30},
                {"kind": "ema", "length": 50},
                # {"kind": "ema", "length": 100},
                {"kind": "ema", "length": 200},
                {"kind": "bbands", "length": 21},
                {"kind": "psar"}
            ]
        )
        df.ta.strategy(CustomStrategy) # To run your "Custom Strategy"
        interval_cumulate_lv(df, cumulate_lv)
    return cumulate_lv

def interval_previous_price_frequency_calc(symbol_, interval):
    interval_previous_price_frequency = []
    df=klines(symbol=symbol_, interval=interval, limit=2)
    if len(df) > 1 :
        low = df['low'][-2]
        high = df['high'][-2]
    else:
        low = df['low'][-1]
        high = df['high'][-1]
    if (high - low) == 0 :
        previous_price_frequency = 0
    else:
        previous_price_frequency = (high/low - 1)*100
    interval_previous_price_frequency.append(previous_price_frequency)
    return interval_previous_price_frequency


def greedy_percentage_calc(interval_peak):
    if interval_peak == '':
        interval_peak = '15m'
    previous_price_frequency = interval_previous_price_frequency_calc(symbol_, interval_peak)[0]
    previous_price_frequency_safe_value = previous_price_frequency*70/100
    if previous_price_frequency_safe_value > (3*taker_fee): # fee 0.08 + 0.08 = 0.16
        greedy_percentage = previous_price_frequency_safe_value
    else:
        greedy_percentage = 3*taker_fee
    return previous_price_frequency, greedy_percentage, interval_peak


def stric_exit_price_calc(position_side, position_entry_price, scalping_direction_pick):
    stric_exit_price = 0
    # interval_peak = globals()['atr_pick']

    # if globals()['volatility_macro_state'] == 1:
    #     interval_peak = globals()['volatility_micro_interval_pick']
    # else:
    #     interval_peak = cpu['info']['itv']

    if scalping_direction_pick == 'neutral':
        interval_peak = '5m'
    elif position_side == 'long' :
        interval_peak = globals()['long_trend_micro_interval_pick']
    elif position_side == 'short':
        interval_peak = globals()['short_trend_micro_interval_pick']
    else:
        interval_peak = ''

    if interval_peak == '':
        interval_peak = '5m'

    ########### 주석처리 필요 ############
    # interval_peak = '15m'
     ########### 주석처리 필요 ############

    greedy_percentage_calc_values = greedy_percentage_calc(interval_peak)
    previous_price_frequency = greedy_percentage_calc_values[0]
    greedy_percentage = greedy_percentage_calc_values[1]
    picked_interval = greedy_percentage_calc_values[2]
    if position_side == 'long' :
        stric_exit_price = position_entry_price*(1 + (taker_fee + greedy_percentage) / 100)
        if stric_exit_price != 0:
            stric_exit_price = float(exchange.price_to_precision(symbol_, stric_exit_price))
    elif position_side == 'short':
        stric_exit_price = position_entry_price*(1 - (taker_fee + greedy_percentage) / 100)
        if stric_exit_price != 0:
            stric_exit_price = float(exchange.price_to_precision(symbol_, stric_exit_price))
    return stric_exit_price, previous_price_frequency, greedy_percentage, picked_interval


def pick_calc(cumulate_lv, position_side, position_size, position_entry_price, symbol_ticker_last, scalping_direction_pick):
    stric_exit_price_values = stric_exit_price_calc(position_side, position_entry_price, scalping_direction_pick)
    stric_exit_price = stric_exit_price_values[0]
    previous_price_frequency = stric_exit_price_values[1]
    greedy_percentage = stric_exit_price_values[2]
    picked_interval = stric_exit_price_values[3]
    # if position_side == 'long' :
    #     atr_exit_price = (position_entry_price*(1 + (2*taker_fee) / 100)) + (atr_given * 5)
    # elif position_side == 'short':
    #     atr_exit_price = (position_entry_price*(1 - (2*taker_fee) / 100)) - (atr_given * 5)
    if position_size == 0:
        pick_min = max([i for i in cumulate_lv if symbol_ticker_last>i], default=symbol_ticker_last) # current_position 이 short 일때 사용
        pick_max = min([i for i in cumulate_lv if symbol_ticker_last<i], default=symbol_ticker_last) # current_position 이 long 일때 사용
        # pick_min = min([i for i in cumulate_lv if symbol_ticker_last>i], default=symbol_ticker_last) # current_position 이 short 일때 사용
        # pick_max = max([i for i in cumulate_lv if symbol_ticker_last<i], default=symbol_ticker_last) # current_position 이 long 일때 사용        
    else:
        pick_min = max([i for i in cumulate_lv if stric_exit_price>=i], default=stric_exit_price) # current_position 이 short 일때 사용
        pick_max = min([i for i in cumulate_lv if stric_exit_price<=i], default=stric_exit_price) # current_position 이 long 일때 사용
        # pick_min = min([i for i in cumulate_lv if stric_exit_price>=i], default=stric_exit_price) # current_position 이 short 일때 사용
        # pick_max = max([i for i in cumulate_lv if stric_exit_price<=i], default=stric_exit_price) # current_position 이 long 일때 사용        
        # if ((globals()['atr_pick']) and (globals()['atr_pick'] in ['1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w'])):
        #    pick_min = max([i for i in cumulate_lv if atr_exit_price>=i], default=pick_min) # current_position 이 short 일때 사용
        #    pick_max = min([i for i in cumulate_lv if atr_exit_price<=i], default=pick_max) # current_position 이 long 일때 사용
    pick_min = float(exchange.price_to_precision(symbol_, pick_min)) # current_position 이 short 일때 사용
    pick_max = float(exchange.price_to_precision(symbol_, pick_max)) # current_position 이 long 일때 사용
    return(pick_max, pick_min, previous_price_frequency, greedy_percentage, picked_interval)

def stric_exit_price_calc_min(position_side, position_entry_price): #수수료만 뺀 본전가 산출
    stric_exit_price = 0
    if position_side == 'long' :
        stric_exit_price = position_entry_price*(1 + ((3*taker_fee) / 100))
        if position_entry_price != 0:
            stric_exit_price = float(exchange.price_to_precision(symbol_, stric_exit_price))
    elif position_side == 'short':
        stric_exit_price = position_entry_price*(1 - ((3*taker_fee) / 100))
        if position_entry_price != 0:        
            stric_exit_price = float(exchange.price_to_precision(symbol_, stric_exit_price))
    return stric_exit_price




def stop_loss_calc(stg_type, success, peaker_side, position_entry_price, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min, stop_loss_range):

    break_even_1 = 0
    break_even_2 = 0
    stop_loss_1 = 0
    stop_loss_2 = 0
    take_profit_1 = 0
    take_profit_2 = 0
    take_profit_3 = 0
    break_even_zero = 0
    max_profit_60percent = 0

    if position_entry_price and position_entry_price != 0:

        if (globals()['atr_pick'] != ''):
            interval_ = globals()['atr_pick']
        else:
            interval_ = '5m'

        # interval_ = '5m'
        stop_loss_range_constant = (globals()['df_' + interval_]['atr_p'].iloc[-1])/100

        # stop_loss_range_constant = 0.003
        break_even_range_constant = stop_loss_range_constant * 2


        entry_time = datetime.datetime.fromtimestamp(position_entry_time_)

        if (position_entry_time_ != 0) and (df_1m['open_time2'] >= entry_time).any():
            df_after_entry = df_1m[df_1m['open_time2'] >= entry_time][['high', 'low']]
            all_time_high = df_after_entry['high'].max()
            all_time_low = df_after_entry['low'].min()            
        else:
            df_after_entry = 0
            all_time_high = 0
            all_time_low = 0



        if peaker_side == 'long':
            current_p_ = symbol_ticker_last

            # break_even = max((last_peaked_price * (1 + break_even_range_constant)), (position_entry_price * (1 + break_even_range_constant)))

            # if l_p_l and l_p_l != 0:
            #     stop_loss = (l_p_l * (1 - stop_loss_range_constant)) # 0.0015 ==  0.15%, 0.15% * 레버리지 배율
            # else:
            #     stop_loss = (position_entry_price * (1 - stop_loss_range_constant))

            # if l_p_h and l_p_h != 0:
            #     take_profit = (l_p_h * (1 + stop_loss_range_constant)) # 0.0015 ==  0.15%, 0.15% * 레버리지 배율
            # else:
            #     take_profit = (position_entry_price * (1 + stop_loss_range_constant))


            # break_even_zero = (position_entry_price * (1 + 0.0012))

            break_even_1 = last_peaked_price * (1 + break_even_range_constant)
            break_even_2 = position_entry_price * (1 + break_even_range_constant)
            # stop_loss_1 = l_p_l * (1 - stop_loss_range_constant)
            stop_loss_2 = position_entry_price * (1 - stop_loss_range_constant)
            # take_profit_1 = l_p_h * (1 + stop_loss_range_constant)
            # take_profit_2 = l_p_l * (1 + stop_loss_range_constant)
            take_profit_3 = position_entry_price * (1 + stop_loss_range_constant)
            break_even_zero = position_entry_price * (1 + 0.0012)

            if (all_time_high > break_even_1) and (break_even_1 > position_entry_price):
                max_profit_60percent = all_time_high - (all_time_high - position_entry_price) * 0.3 # long
                # max_profit_60percent = all_time_low + (position_entry_price - all_time_low) * 0.4 # short
            else:
                max_profit_60percent = 0

            # 리스트 정렬
            values = [break_even_1, break_even_2, stop_loss_1, stop_loss_2, take_profit_1, take_profit_2, take_profit_3, break_even_zero, max_profit_60percent, stric_exit_min_price]
            filtered_values = [value for value in values if value != 0] # 0 제외
            sorted_values = sorted(filtered_values, reverse=True)

            # 초기 stop_loss 값 설정 (기본값은 최소값)
            stop_loss = sorted_values[-1]

            # 조건문을 통해 stop_loss 결정
            for value in sorted_values:
                if current_p_ > value:
                    stop_loss = value
                    break
            
            print(f' interval_:{interval_}, current_p_:{current_p_}, stric_exit_min_price:{stric_exit_min_price}, stop_loss_range_constant:{stop_loss_range_constant}, break_even_range_constant:{break_even_range_constant}, break_even:{break_even_2}, l_p_l:{l_p_l}, stop_loss:{stop_loss}')

            stop_loss_ = float(exchange.price_to_precision(symbol_, stop_loss)) # current_position 이 short 일때 사용
            percentage_difference_stop_loss_ = ((stop_loss_ - current_p_) / current_p_) * 100
            target_p_ = pick_max
            percentage_difference_target_p_ = ((target_p_ - current_p_) / current_p_) * 100

        elif peaker_side == 'short':
            current_p_ = symbol_ticker_last

            # break_even = min((last_peaked_price * (1 - break_even_range_constant)), (position_entry_price * (1 - break_even_range_constant)))

            # if l_p_h and l_p_h != 0:
            #     stop_loss = (l_p_h * (1 + stop_loss_range_constant))
            # else:
            #     stop_loss = (position_entry_price * (1 + stop_loss_range_constant))

            # if l_p_l and l_p_l != 0:
            #     take_profit = (l_p_l * (1 - stop_loss_range_constant)) # 0.0015 ==  0.15%, 0.15% * 레버리지 배율
            # else:
            #     take_profit = (position_entry_price * (1 - stop_loss_range_constant))

            # break_even_zero = (position_entry_price * (1 - 0.0012))

            break_even_1 = last_peaked_price * (1 - break_even_range_constant)
            break_even_2 = position_entry_price * (1 - break_even_range_constant)
            # stop_loss_1 = l_p_h * (1 + stop_loss_range_constant)
            stop_loss_2 = position_entry_price * (1 + stop_loss_range_constant)
            # take_profit_1 = l_p_l * (1 - stop_loss_range_constant)
            # take_profit_2 = l_p_h * (1 - stop_loss_range_constant)
            take_profit_3 = position_entry_price * (1 - stop_loss_range_constant)
            break_even_zero = position_entry_price * (1 - 0.0012)

            if (all_time_low < break_even_1) and (break_even_1 < position_entry_price):
                # max_profit_60percent = all_time_high - (all_time_high - position_entry_price) * 0.4 # long
                max_profit_60percent = all_time_low + (position_entry_price - all_time_low) * 0.3 # short
            else:
                max_profit_60percent = 0

            # 리스트 정렬
            values = [break_even_1, break_even_2, stop_loss_1, stop_loss_2, take_profit_1, take_profit_2, take_profit_3, break_even_zero, max_profit_60percent, stric_exit_min_price]
            filtered_values = [value for value in values if value != 0] # 0 제외
            sorted_values = sorted(filtered_values, reverse=False)

            # 초기 stop_loss 값 설정 (기본값은 최소값)
            stop_loss = sorted_values[-1]

            # 조건문을 통해 stop_loss 결정
            for value in sorted_values:
                if current_p_ < value:
                    stop_loss = value
                    break

            print(f' interval_:{interval_}, current_p_:{current_p_}, stric_exit_min_price:{stric_exit_min_price}, stop_loss_range_constant:{stop_loss_range_constant}, break_even_range_constant:{break_even_range_constant}, break_even:{break_even_2}, l_p_l:{l_p_l}, stop_loss:{stop_loss}')
            stop_loss_ = float(exchange.price_to_precision(symbol_, stop_loss)) # current_position 이 short 일때 사용
            percentage_difference_stop_loss_ = ((current_p_ - stop_loss_) / current_p_) * 100
            target_p_ = pick_min
            percentage_difference_target_p_ = ((current_p_ - target_p_) / current_p_) * 100

            print('---------------------------')
            print(f'break_even: {break_even_2}')
            print('---------------------------')
            # print(stg_type, success, peaker_side, symbol_ticker_last, close_price_low, close_price_high, pick_max, pick_min)
            # print(stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_)
    else:
        stop_loss_ = percentage_difference_stop_loss_ = target_p_ = percentage_difference_target_p_ = 0

    return stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_




def exit_order_position_amount_calc(position_size):
    if position_size == float(0):
        exit_order_position_amount = 0
    else:
        exit_order_position_amount = float(exchange.amount_to_precision(symbol_, position_size))
        if exchange_id == 'huobi':
            exit_order_position_amount = float(exchange.amount_to_precision(symbol_, (position_size * 1000)))
    return exit_order_position_amount




def get_symbols_from_processes(running_markets):
    # ps -ef 명령어 실행하여 모든 프로세스 목록 가져오기
    result = subprocess.run(['ps', '-ef'], stdout=subprocess.PIPE, text=True)
    process_list = result.stdout.splitlines()

    symbols = []

    # 각 프로세스 라인에서 필요한 정보 추출
    for line in process_list:
        for script_name in running_markets:
            if script_name in line:
                # 정규식을 사용하여 스크립트 이름 다음에 오는 심볼 추출
                pattern = rf"{re.escape(script_name)}\s+(\S+)"
                match = re.search(pattern, line)
                if match:
                    symbol = match.group(1)
                    symbols.append(symbol)
    symbols = set(symbols)
    return symbols


























# cpu = get_configuration(trader_name, )
# api_key = cpu['info']['api_key']
# api_secret = cpu['info']['api_secret']
# exchange_id = cpu['exchange_id']

api_key = 'aiJSPacDRzVVEobWHmiHjRdVLdh6725kR6iKqSgEofU9PpZehmvHubh86uHNiSkc'
api_secret = 'O8Q2yh2gfdLkxhv3LCIc452Yo2WtH2NVXH9yt7j7Jsse6QYzNv3kliGMw7axMso7'
exchange_id = 'binanceusdm'

exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'options': {
        #'defaultType': 'future',
        'adjustForTimeDifference': True,
    },
})

if exchange_id == 'bybit':
    params = {
        'position_idx': 0,
    }
else:
    params = ''



# params = {
#     'recvWindow': 60000  # 기본값은 5000ms
# }

exchange.load_time_difference()
markets = exchange.load_markets()

running_markets = {
    'live3_4.py': '',
    'live3_5.py': '',
    'live3_6.py': '',
    'live3_7.py': '',
    # 'live3_8.py': ''
    # 'live3_9.py': ''
}

base_markets = {
    'live3_4.py': '',
    'live3_5.py': '',
    'live3_6.py': '',
    'live3_7.py': '',
    # 'live3_8.py': ''
    # 'live3_9.py': ''
}

# intervals = ['5m','15m', '1h']
# intervals_ = ['15m', '1h']
intervals_2 = ['1m', '5m', '15m', '1h']
intervals_ = ['15m', '1h']
intervals = ['1h']
a = 0
balance_currency = 'USDT'
limit = 1000
taker_fee = 0.12
globals()['long_trend_micro_interval_pick'] = ''
globals()['short_trend_micro_interval_pick'] = ''
stopPrice_const = 2
globals()['atr_pick'] = ''



while True:
    a += 1
    j = 0
    start = time.time()
    top_markets = {}
    current_max_changes = {}
    out_ranked_markets = {}
    new_ranked_markets = {}
    available_slots = {}
    markets_to_start = {}
    wallet_balance = wallet_balance_fix = balance_calc(exchange_id, balance_currency)
    

    symbols_list = get_symbols_from_processes(running_markets)
    open_positions = exchange.fetch_positions()
    open_positions_count = len(open_positions)

    print(symbols_list)
    for position in open_positions:
        # print(position)
        # print('\n')
        unrealised_pnl_ = position['unrealizedPnl']
        position_side_ = position['side']
        symbol_ = position['symbol'] # TON/USDT:USDT
        symbol_2 = position['info']['symbol'] # TONUSDT
        position_size_ = position['contracts']
        position_entry_price_ = position["entryPrice"]
        position_entry_time_ = position["timestamp"]/1000
        
        if symbol_2 not in symbols_list:
            print('not in', symbol_2)
            # 손실이 지갑의 3% 초과 시 반대진입
            # if (unrealised_pnl_ / wallet_balance) * 100 <= -5:  # 손실이 3% 초과 시
            #     exchange.cancel_all_orders(symbol_)
            #     print(f"{symbol_}: all open orders canceled")
                
            #     if position_side_ == 'long':
            #         exit_side = 'sell'
            #         limit_type_only_case_price_pick_param = float(1 - 0.0005)
            #     elif position_side_ == 'short':
            #         exit_side = 'buy'
            #         limit_type_only_case_price_pick_param = float(1 + 0.0005)

            #     print(f"{symbol_}: 손실이 지갑의 -0.5% 초과 하여 반대진입")
            #     print(exit_side)
            #     print(position_size_*2)
            #     # 반대진입, 포지션 크기의 2배로 진입
            #     exchange.create_order(symbol=symbol_, type='market', side=exit_side, amount=position_size_*2, params={})
                
            #     send_to_telegram(f"{symbol_}: 손실이 지갑의 -0.5% 초과 하여 반대진입")
    




            if ((unrealised_pnl_ / wallet_balance) * 100 >= 2) or ((unrealised_pnl_ / wallet_balance) * 100 < -2):  # 이익이 3% 이상이거나 손실이 3% 이하인 경우

                exchange.cancel_all_orders(symbol_)
                print(f"{symbol_}: all open orders canceled")
                
                if position_side_ == 'long':
                    exit_side = 'sell'
                    limit_type_only_case_price_pick_param = float(1 - 0.0005)
                elif position_side_ == 'short':
                    exit_side = 'buy'
                    limit_type_only_case_price_pick_param = float(1 + 0.0005)

                print(f"{symbol_}: 이익이 지갑의 2% 초과, 익절")
                print(exit_side)
                print(position_size_)
                # 반대진입, 포지션 크기의 2배로 진입
                exchange.create_order(symbol=symbol_, type='market', side=exit_side, amount=position_size_, params={'reduceOnly': True})
                
                send_to_telegram(f"{symbol_}: 이익이 지갑의 2% 초과, 익절")

        else:
            print('in', symbol_2)







    end = time.time()
    total_time = end - start
    print(str(total_time/60) + ' min')
    
    formatted_datetime = datetime.datetime.now().replace(microsecond=0)
    print(formatted_datetime)
    print('\n')
    # if len(running_markets) < 3:
    #     time.sleep(30)
    # else:
    #     time.sleep(180)
    time.sleep(180)