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
import subprocess, psutil
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
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

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
        if(df['close'].count() > 600):
            CustomStrategy = ta.Strategy( # Create your own Custom Strategy
                name="Momo and Volatility",
                description="RSI, MACD, adx, dmp, dmn, obv, atr",
                ta=[
                    {"kind": "adx"},
                    {"kind": "adx", "length": 200, "lensig": 14},
                    {"kind": "bbands", "length": 21},
                    {"kind": "atr"},
                    {"kind": "atr", "length": 1},
                    {"kind": "atr", "length": 200},



                    {"kind": "kdj"},
                    {"kind": "rsi"},
                    {"kind": "rsi", "length": 200},
                    {"kind": "macd"},
                    {"kind": "macd", "fast": 50, "slow": 75, "signal":35},
                    {"kind": "ema", "length": 200}
                ]
            )

            df.ta.strategy(CustomStrategy) # To run your "Custom Strategy"
            df['adx_diff'] = df['ADX_14'].diff()
            df['dx_diff'] = df['DX_14'].diff()
            df['adx_200_diff'] = df['ADX_200'].diff()
            df['dx_200_diff'] = df['DX_200'].diff()

            df['atr_diff'] = df['ATRr_14'].diff()
            df['atr_p'] = df['ATRr_14']/df['close']*100
            df['atr_p_diff'] = df['atr_p'].diff()

            df['atr200_diff'] = df['ATRr_200'].diff()
            df['atr200_p'] = df['ATRr_200']/df['close']*100
            df['atr200_p_diff'] = df['atr200_p'].diff()

            df['bbb_diff'] = df['BBB_21_2.0'].diff()





            ########################################################################################
            
            df['abs_EMA_200_close_diff'] = abs(df['close'] - df['EMA_200'])
            df['abs_EMA_200_BBU_21_2_diff'] = abs(df['BBU_21_2.0'] - df['EMA_200'])
            df['abs_EMA_200_BBL_21_2_diff'] = abs(df['BBL_21_2.0'] - df['EMA_200'])
            df['abs_dmp_dmn_200_diff'] = abs(df['DMP_200'] - df['DMN_200'])
            df['abs_MACD_50_75_35'] = abs(df['MACD_50_75_35'])
            df['abs_MACD_12_26_9'] = abs(df['MACD_12_26_9'])
            df['abs_RSI_14'] = abs(df['RSI_14'] - 50)
            df['abs_RSI_200'] = abs(df['RSI_200'] - 50)
            mid_value_J = 50  # 중간값
            df['abs_J_9_3'] = abs(df['J_9_3'] - mid_value_J)  # 중간값 기준 절대값 처리

            # 1. 데이터 정규화 (Min-Max Scaling)
            # scaler = MinMaxScaler()
            scaler = MinMaxScaler(feature_range=(0, 1))  # 최대값을 1로 설정

            # 정규화할 열들
            scaling_columns = [
                'abs_EMA_200_close_diff', 
                'abs_EMA_200_BBU_21_2_diff', 
                'abs_EMA_200_BBL_21_2_diff', 
                'abs_dmp_dmn_200_diff', 
                'abs_MACD_50_75_35',
                'atr_p'
                # 'abs_MACD_12_26_9'
            ]

            # 2. 원본 데이터는 유지하고, 정규화된 데이터를 새로운 열에 저장
            df[[f'scaled_{col}' for col in scaling_columns]] = scaler.fit_transform(df[scaling_columns])

            # df['scaled_abs_dmp_dmn_200_diff'] = df['abs_dmp_dmn_200_diff'] / 100.0 * 5.0 # 0 ~ 1로 정규화
            # df['scaled_atr_p'] = df['atr_p'] / 100.0 * 100.0 # 0 ~ 1로 정규화
            # df['scaled_RSI_14'] = df['abs_RSI_14']  / 50.0  # 0 ~ 1로 정규화
            df['scaled_RSI_200'] = df['abs_RSI_200']  / 50.0 # * 5.0 # 0 ~ 1로 정규화
            df['scaled_J_9_3'] = df['abs_J_9_3'] / 250.0  # 0 ~ 1로 정규화

            # 3. 세 가지 비슷한 특성의 비중 조정 (1로 합쳐서 계산)
            df['combined_EMA'] = (
                df['scaled_abs_EMA_200_close_diff'] + 
                df['scaled_abs_EMA_200_BBU_21_2_diff'] + 
                df['scaled_abs_EMA_200_BBL_21_2_diff']
            ) / 3  # 평균으로 비중 조정

            # 4. 나머지 특성들과 combined_EMA를 합쳐서 combined_diff 생성
            df['combined_diff'] = (
                df['combined_EMA'] +  # 조정된 combined_EMA
                df['scaled_abs_dmp_dmn_200_diff'] +
                df['scaled_abs_MACD_50_75_35'] +
                # df['scaled_abs_MACD_12_26_9'] +

                df['scaled_atr_p'] +
                # df['scaled_RSI_14'] +
                df['scaled_RSI_200']
                # df['scaled_J_9_3']
                
            ) / 5  # 평균으로 비중 조정



            window_length = 51  # 윈도우 크기는 홀수여야 합니다.
            polyorder = 2
            df['combined_diff_filtered'] = savgol_filter(df['combined_diff'], window_length=window_length, polyorder=polyorder)
            
            
            df['combined_diff_diff'] = df['combined_diff'].diff()
            df['combined_diff_filtered_diff'] = df['combined_diff_filtered'].diff()


            ########################################################################################


            ############## 2. ##########################################################################

            df['second_EMA_200_close_diff'] = df['close'] - df['EMA_200']
            df['second_MACD_50_75_35'] = df['MACD_50_75_35']
            df['second_dmp_dmn_200_diff'] = df['DMP_200'] - df['DMN_200']
            df['second_RSI_200'] = (df['RSI_200'] - 50)/50
            
            # 1. 데이터 정규화 (Min-Max Scaling)
            # scaler = MinMaxScaler()
            scaler = MinMaxScaler(feature_range=(-1, 1))  # 최대값을 1로 설정

            # 정규화할 열들
            scaling_columns = [
                'second_EMA_200_close_diff',
                'second_MACD_50_75_35',
                'second_dmp_dmn_200_diff'
            ]

            # 2. 원본 데이터는 유지하고, 정규화된 데이터를 새로운 열에 저장
            df[[f'scaled_{col}' for col in scaling_columns]] = scaler.fit_transform(df[scaling_columns])

            # df['scaled_second_dmp_dmn_200_diff'] = df['second_dmp_dmn_200_diff'] / 100.0 * 4.0  # 0 ~ 1로 정규화
            df['scaled_second_RSI_200'] = df['second_RSI_200']


            # 4. 나머지 특성들과 combined_EMA를 합쳐서 combined_diff 생성
            df['second_combined_diff'] = (
                df['scaled_second_EMA_200_close_diff'] +  # 조정된 combined_EMA
                df['scaled_second_MACD_50_75_35'] +
                df['scaled_second_dmp_dmn_200_diff'] +
                df['scaled_second_RSI_200']
            ) / 4  # 평균으로 비중 조정


            window_length = 51  # 윈도우 크기는 홀수여야 합니다.
            polyorder = 2
            df['second_combined_diff_filtered'] = savgol_filter(df['second_combined_diff'], window_length=window_length, polyorder=polyorder)
            
            df['second_combined_diff_diff'] = df['second_combined_diff'].diff()
            df['second_combined_diff_filtered_diff'] = df['second_combined_diff_filtered'].diff()

            ########################################################################################






#                 ########################################################################################################################################################
# ########################################################################################################################################################
# ########################################################################################################################################################
# ########################################################################################################################################################
# ########################################################################################################################################################
#                 ###################################################################################################
#                 ###################################################################################################



#             iqr_close = None
#             iqr_feature1 = None
#             iqr_feature2 = None

#             df['feature1'] = 0
#             df['feature2'] = 0

#             df['anomalies_close'] = False
#             df['anomalies_feature1'] = False
#             df['anomalies_feature2'] = False

#             df['maxima_peak_x_anomalies_close'] = 0
#             df['minima_peak_x_anomalies_close'] = 0
#             df['maxima_peak_x_anomalies_feature1'] = 0
#             df['minima_peak_x_anomalies_feature1'] = 0
#             df['maxima_peak_x_anomalies_feature2'] = 0
#             df['minima_peak_x_anomalies_feature2'] = 0



#             df_cleaned = df[[ \

#                 'BBB_21_2.0', \
#                 'volume', \
#                 'ATRr_1', \
                





                
#             ]].select_dtypes(include=np.number).dropna()


#             # # 데이터 준비
#             # columns = df_cleaned.columns
#             # X = df_cleaned[columns].values

#             # 데이터 준비
#             columns = df_cleaned.columns
#             # X = df_cleaned[columns].values
#             desired_columns = ['BBB_21_2.0', 'volume', 'ATRr_1']
#             # desired_columns = ['ATRr_1', 'ATRr_14']
#             X = df_cleaned[desired_columns].values
            




#             # 데이터 전처리
#             scaler = StandardScaler()
#             scaled_data = scaler.fit_transform(X)

#             # 저장된 모델 불러오기
#             # print(interval)
#             loaded_autoencoder = load_model("model_1.h5")
            
#             # loaded_autoencoder.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
#             # Compile the model (assuming you have defined loss and optimizer during training)


#             # 학습된 오토인코더를 사용하여 데이터를 변환
#             # 수정된 부분: scaled_data 대신에 새로운 데이터를 사용하여 변환
#             encoded_data_mlp = loaded_autoencoder.predict(scaled_data)  # 수정된 부분: 저장된 모델을 사용하여 데이터 변환

#             # PCA를 사용하여 데이터를 2차원으로 축소
#             pca = PCA(n_components=2)
#             encoded_data_2d = pca.fit_transform(encoded_data_mlp)

#             df.loc[df_cleaned.index, 'feature1'] = encoded_data_2d[:, 0]  # df_cleaned의 인덱스를 기준으로 값 할당
#             df.loc[df_cleaned.index, 'feature2'] = encoded_data_2d[:, 1]

#             df['feature1_diff'] = df['feature1'].diff()
#             df['feature2_diff'] = df['feature2'].diff()
#             df['feature1_percentage_change'] = np.where(df['feature1_diff'].abs().shift() != 0, (df['feature1_diff'].diff() / df['feature1_diff'].abs().shift()) * 100, 0)
#             df['feature2_percentage_change'] = np.where(df['feature2_diff'].abs().shift() != 0, (df['feature2_diff'].diff() / df['feature2_diff'].abs().shift()) * 100, 0)

#                 ########################################################################################################################################################
# ########################################################################################################################################################
# ########################################################################################################################################################
# ########################################################################################################################################################
# ########################################################################################################################################################
#                 ###################################################################################################
#                 ###################################################################################################




















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
    if df['close'].count() > 1 :
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



def calculate_predicted_change(symbol_):
    # 1. 데이터 가져오기
    df = klines(symbol=symbol_, interval='1d', limit=2)

    # 2. 전날 데이터
    low_yesterday = df['low'][-2]
    high_yesterday = df['high'][-2]
    Volume_yesterday = df['volume'][-2]
    Change_yesterday = ((high_yesterday - low_yesterday) / low_yesterday) * 100  # 전날 변동성 (%)
    C_yesterday = Change_yesterday / Volume_yesterday  # 전날 변동 상수

    # 3. 오늘 데이터 (현재까지)
    low_today = df['low'][-1]
    high_today = df['high'][-1]
    Volume_today_so_far = df['volume'][-1]
    Change_today_so_far = ((high_today - low_today) / low_today) * 100  # 현재까지의 변동성 (%)
    C_today = Change_today_so_far / Volume_today_so_far  # 현재까지의 변동 상수

    ##########################################################################################
    # # 4. 오늘 최종 거래량 (예측 또는 마감 시점 데이터)
    # # 현재 시간을 UTC 기준으로 설정
    # current_time_utc = datetime.datetime.utcnow()  # UTC 기준 현재 시간

    # # CME 정규 거래 시작 시간 (UTC)
    # cme_open_time_utc = current_time_utc.replace(hour=14, minute=30, second=0, microsecond=0)

    # # 만약 시간이 이미 시작 시간 이전이라면, 남은 시간을 24시간 기준으로 설정
    # if current_time_utc < cme_open_time_utc:
    #     remaining_time_utc = (cme_open_time_utc - current_time_utc).total_seconds() / 3600  # 남은 시간 (단위: 시간)
    # else:
    #     remaining_time_utc = 24 - (current_time_utc - cme_open_time_utc).total_seconds() / 3600  # 남은 시간 (24시간 기준)

    ##########################################################################################
    # 5. 오늘 최종 거래량 (예측 또는 마감 시점 데이터)
    # 현재 시간을 UTC 기준으로 설정
    current_time_utc = datetime.datetime.utcnow()  # UTC 기준 현재 시간
    midnight_utc = current_time_utc.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)  # 다음날 UTC 자정 시간
    remaining_time_utc = (midnight_utc - current_time_utc).total_seconds() / 3600  # 남은 시간 (단위: 시간)
    ##########################################################################################

    # 거래량 예측: 현재까지 거래량 * 남은 시간 비율로 조정
    # Volume_today_final = Volume_today_so_far * (1 + remaining_time_utc / 24)  # 남은 시간을 반영한 거래량 예측
    Volume_today_final = (Volume_today_so_far * 24) / (24 - remaining_time_utc)

    ##########################################################################################
    # # 6. 오늘 최종 변동성 예측
    Predicted_Change_today = C_today * Volume_today_final
    print(C_today)
    print(Predicted_Change_today)



    # Volume_prediction_1 = Change_yesterday/C_today
    # Predicted_Change_today = Volume_prediction_1 * C_today
    ##########################################################################################

    return Predicted_Change_today




























# cpu = get_configuration(trader_name, )
# api_key = cpu['info']['api_key']
# api_secret = cpu['info']['api_secret']
# exchange_id = cpu['exchange_id']

api_key = 'zVfLUNtbPLsXCLxyKCIMi3TFLeWRUZcWem3PbRaHXpAYvXU0JZo63oaQ6CVxLBhN'
api_secret = 'hFXLLKXwgBZuh16H3JpLngPgxbAZHzEbYA9rqK7RN9VovS2JgPSzOosaeMzeobz7'
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

markets = exchange.load_markets()

running_markets = {
    '/aws/engin1/live3_34.py': '',
    '/aws/engin1/live3_35.py': '',
    '/aws/engin1/live3_36.py': '',
    '/aws/engin1/live3_37.py': ''
    # 'live3_8.py': ''
    # 'live3_9.py': ''
}

base_markets = {
    '/aws/engin1/live3_34.py': '',
    '/aws/engin1/live3_35.py': '',
    '/aws/engin1/live3_36.py': '',
    '/aws/engin1/live3_37.py': ''
    # 'live3_8.py': ''
    # 'live3_9.py': ''
}

# intervals = ['5m','15m', '1h']
# intervals_ = ['15m', '1h']
intervals_2 = ['1m', '5m', '15m', '1h', '4h']
# intervals_ = ['1h', '4h']
# intervals = ['4h']
intervals_ = ['5m', '15m', '1h']
intervals = ['15m']
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
    # wallet_balance = wallet_balance_fix = balance_calc(exchange_id, balance_currency)

    exchange.load_time_difference()
    open_positions = exchange.fetch_positions()
    open_positions_count = len(open_positions)

    if open_positions_count < 12:

        for market in markets:
            if (':USDT' in market): # and ('YFIUSDT' not in market) and ('LISTAUSDT' not in market):
                j += 1
                symbol = pair = markets[market]['id']
                current_time = dt.datetime.now()
                print(f'\n#{a}-{j}:[{market}]\n')
                condition = 0
                percentage_change = 0
                for b in intervals_2:
                    globals()['df_' + b] = pd.DataFrame()

                try:
                    # ticker_data = exchange.fetch_ticker(symbol)
                    # percentage_change = ticker_data['percentage']

                    predicted_change = calculate_predicted_change(symbol)



                    ohlcv_data = exchange.fetch_ohlcv(symbol=symbol, timeframe='1d', limit=1)
                    open_price = ohlcv_data[0][1]  # Open price is the second element in the first data point
                    last_price = ohlcv_data[0][4]  # Last price is the fifth element in the first data point
                    percentage_change = ((last_price - open_price) / open_price) * 100

                    peak_calc(symbol, intervals_, 1000)


                    print(
                        (
                            (globals()['df_1h']['ADX_200'].count() > 250)
                                and
                            ((globals()['df_15m']['combined_diff_filtered_diff'].iloc[-1]) > 0)
                            #     and
                            # (predicted_change < 30)



                            #     and
                            # ((globals()['df_4h']['combined_diff_filtered'].iloc[-1]) > 0.25)
                                and
                            (
                                (
                                    ((globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]) > 0.3)
                                    and (globals()['df_5m']['second_combined_diff_filtered'].iloc[-1] > 0.3)
                                )
                                    or
                                (
                                    ((globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]) < -0.3)
                                    and (globals()['df_5m']['second_combined_diff_filtered'].iloc[-1] < -0.3)
                                )
                            )

                            #     and
                            # ((globals()['df_15m']['combined_diff_filtered'].iloc[-1]) < 0.3)
                            #     and
                            # ((globals()['df_5m']['combined_diff_filtered'].iloc[-1]) < 0.3)
                        )
                    )

                    

                    # dmp_dmn_diff = (globals()['df_15m']['DMP_200'] - globals()['df_15m']['DMN_200']).diff()
                    # print(abs(dmp_dmn_diff.iloc[-1]))
                    # print(abs(globals()['df_4h']['dx_200_diff'].iloc[-1]))

                    # print(globals()['df_15m']['combined_diff'].iloc[-1])
                    # print(globals()['df_15m']['combined_diff_diff'].iloc[-1])



                    for interval in intervals:
                        # print(f"[feature1: {globals()['df_'+interval]['feature1'].iloc[-1]}]")


                        # print('interval:', 'df_'+interval)
                        # print('count:', globals()['df_'+interval]['ADX_200'].count())
                        # print('ADX_200:', (globals()['df_'+interval]['ADX_200'].iloc[-1]))
                        # print('ADX_14:', (globals()['df_'+interval]['ADX_14'].iloc[-1]))
                        # print('top_markets:', top_markets)
                        # print('percentage_change:', percentage_change)

                        condition = (
                            # ((globals()['df_'+interval]['atr_p_diff'].iloc[-1]) > 0) 
                            # and 
                            ###############################################################
                            # ((globals()['df_'+interval]['adx_diff'].iloc[-1]) > 0)
                            # and ((globals()['df_'+interval]['ADX_14'].iloc[-1]) > 25)
                            # and 


                            ###############################################################
                            # ((globals()['df_'+interval]['feature1'].iloc[-1]) > 0)
                            # and ((globals()['df_'+interval]['feature1_diff'].iloc[-1]) > 0)
                            ###############################################################
                            # ((globals()['df_'+interval]['adx_diff'].iloc[-1]) < 0)
                            # and ((globals()['df_'+interval]['ADX_14'].iloc[-1]) < 25)
                            # and 


                            # ###############################################################
                            # # 1.


                            # (globals()['df_4h']['ADX_200'].count() > 500)
                            # and
                            # ((globals()['df_4h']['atr200_p_diff'].iloc[-1]) < 0)
                            # and                        
                            # ((globals()['df_4h']['adx_200_diff'].iloc[-1]) < 0)
                            # and
                            # ((globals()['df_4h']['bbb_diff'].iloc[-1]) <= 0)

                            # # and
                            # # ((globals()['df_4h']['atr200_p'].iloc[-1]) < 2)
                            
                            # and
                            # ((globals()['df_4h']['ADX_200'].iloc[-1]) <= 5)
                            # and
                            # ((globals()['df_4h']['feature1'].iloc[-1]) <= -0.2)

                            # # and
                            # # ((globals()['df_4h']['feature1'].iloc[-1]) > 0)
                            
                            # ###############################################################


                            # 2.


                            # (globals()['df_4h']['ADX_200'].count() > 500)
                            # # and
                            # # ((globals()['df_4h']['atr200_p_diff'].iloc[-1]) < 0)
                            # # and                        
                            # # ((globals()['df_4h']['adx_200_diff'].iloc[-1]) < 0)
                            # # and
                            # # ((globals()['df_4h']['bbb_diff'].iloc[-1]) <= 0)

                            # # and
                            # # ((globals()['df_4h']['atr200_p'].iloc[-1]) < 2)
                            
                            # and
                            # ((globals()['df_4h']['ADX_200'].iloc[-1]) < 8)
                            # and
                            # ((globals()['df_4h']['feature1'].iloc[-1]) <= -0.2)

                            # # and
                            # # ((globals()['df_4h']['feature1'].iloc[-1]) > 0)
                            


                            # # 3.
                            # # (globals()['df_4h']['ADX_200'].count() > 500)
                            # # and
                            # (abs(globals()['df_15m']['DMP_200'].iloc[-1] - globals()['df_15m']['DMN_200'].iloc[-1]) > 2.7)
                            # and
                            # (abs(globals()['df_15m']['DX_200'].iloc[-1] - globals()['df_15m']['ADX_200'].iloc[-1]) > 1.8)


                            # 4.
                            # ((globals()['df_15m']['combined_diff'].iloc[-1]) > 0.4)
                            # and
                            # (
                            #     (globals()['df_15m']['ADX_200'].count() > 500)
                            #     and
                            #     ((globals()['df_15m']['combined_diff_filtered'].iloc[-1]) < 0.2)
                            #     and
                            #     ((globals()['df_5m']['combined_diff_filtered'].iloc[-1]) < 0.2)
                            # )
                            # or
                            # (
                            #     (globals()['df_15m']['ADX_200'].count() > 500)
                            #     and
                            #     ((globals()['df_15m']['combined_diff_filtered'].iloc[-1]) > 0.5)
                            #     and
                            #     ((globals()['df_5m']['combined_diff_filtered'].iloc[-1]) > 0.5)
                            # )

                            (
                                ###############################################################
                                # 1.

                                (
                                    (globals()['df_1h']['ADX_200'].count() > 250)
                                        and
                                    ((globals()['df_15m']['combined_diff_filtered_diff'].iloc[-1]) > 0)
                                    #     and
                                    # (predicted_change < 30)



                                    #     and
                                    # ((globals()['df_4h']['combined_diff_filtered'].iloc[-1]) > 0.25)
                                        and
                                    (
                                        (
                                            ((globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]) > 0.3)
                                            and (globals()['df_5m']['second_combined_diff_filtered'].iloc[-1] > 0.3)
                                        )
                                            or
                                        (
                                            ((globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]) < -0.3)
                                            and (globals()['df_5m']['second_combined_diff_filtered'].iloc[-1] < -0.3)
                                        )
                                    )

                                    #     and
                                    # ((globals()['df_15m']['combined_diff_filtered'].iloc[-1]) < 0.3)
                                    #     and
                                    # ((globals()['df_5m']['combined_diff_filtered'].iloc[-1]) < 0.3)
                                )


                                # # 2.

                                # (
                                #     (globals()['df_15m']['ADX_200'].count() > 500)
                                #         and
                                #     ((globals()['df_15m']['combined_diff_filtered_diff'].iloc[-1]) > 0)
                                #         and
                                #     ((globals()['df_15m']['combined_diff_filtered'].iloc[-1]) < 0.2)
                                #         and
                                #     (
                                #         (
                                #             ((globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]) < 0.35)
                                #             and (globals()['df_5m']['second_combined_diff_filtered'].iloc[-1] < 0.35)
                                #         )
                                #             or
                                #         (
                                #             ((globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]) > -0.35)
                                #             and (globals()['df_5m']['second_combined_diff_filtered'].iloc[-1] > -0.35)
                                #         )
                                #     )

                                #     #     and
                                #     # ((globals()['df_15m']['combined_diff_filtered'].iloc[-1]) < 0.3)
                                #     #     and
                                #     # ((globals()['df_5m']['combined_diff_filtered'].iloc[-1]) < 0.3)
                                # )


                                ###############################################################
                                # # 3.


                                # (
                                #     (globals()['df_4h']['ADX_200'].count() > 500)
                                #     #     and
                                #     # ((globals()['df_1h']['combined_diff_filtered_diff'].iloc[-1]) > 0)
                                #     # #     and
                                #     # ((globals()['df_4h']['combined_diff_filtered'].iloc[-1]) > 0.25)
                                #         and
                                #     (
                                #         (
                                #             ((globals()['df_4h']['second_combined_diff_filtered'].iloc[-1]) < -0.12 )
                                #             and ((globals()['df_4h']['second_combined_diff_filtered_diff'].iloc[-1]) > 0 )
                                #             and (globals()['df_1h']['second_combined_diff_filtered'].iloc[-1] < -0.2)
                                #             and (globals()['df_1h']['second_combined_diff_filtered_diff'].iloc[-1] > 0)
                                #         )
                                #         #     or
                                #         # (
                                #         #     ((globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]) > -0.35)
                                #         #     and (globals()['df_5m']['second_combined_diff_filtered'].iloc[-1] > -0.35)
                                #         # )
                                #     )

                                #     #     and
                                #     # ((globals()['df_15m']['combined_diff_filtered'].iloc[-1]) < 0.3)
                                #     #     and
                                #     # ((globals()['df_5m']['combined_diff_filtered'].iloc[-1]) < 0.3)
                                # )







                                ###############################################################


                                #     and
                                # (
                                #     (
                                #         ((globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]) > 0.3)
                                #             and
                                #         ((globals()['df_5m']['second_combined_diff_filtered'].iloc[-1]) > 0.3)
                                #         #     and
                                #         # ((globals()['df_15m']['second_combined_diff_filtered_diff'].iloc[-1]) < 0)
                                #         #     and
                                #         # ((globals()['df_5m']['second_combined_diff_filtered_diff'].iloc[-1]) < 0)
                                #     )
                                #         or
                                #     (
                                #         ((globals()['df_15m']['second_combined_diff_filtered'].iloc[-1]) < -0.3)
                                #             and
                                #         ((globals()['df_5m']['second_combined_diff_filtered'].iloc[-1]) < -0.3)
                                #         #     and
                                #         # ((globals()['df_15m']['second_combined_diff_filtered_diff'].iloc[-1]) > 0)
                                #         #     and
                                #         # ((globals()['df_5m']['second_combined_diff_filtered_diff'].iloc[-1]) > 0)
                                #     )
                                # )
                                # and
                                # ((globals()['df_15m']['combined_diff_filtered'].iloc[-1]) < 0.9)
                                # and
                                # ((globals()['df_5m']['combined_diff_filtered'].iloc[-1]) < 0.9)

                                # and
                                # ((globals()['df_15m']['combined_diff_filtered_diff'].iloc[-1]) > 0)
                                # and
                                # ((globals()['df_5m']['combined_diff_filtered_diff'].iloc[-1]) > 0)
                            )

                            # and
                            # ((globals()['df_15m']['combined_diff_diff'].iloc[-1]) > 0)
                            # and
                            # ((globals()['df_15m']['combined_diff_filtered_diff'].iloc[-1]) > 0)
                            # # 5.
                            # ((globals()['df_15m']['combined_diff'].iloc[-1]) < 0.2)


                            # and

                            # 15m
                            # (abs(dmp_dmn_diff.iloc[-1]) > 0.55)
                            # and
                            # (abs(globals()['df_15m']['dx_200_diff'].iloc[-1]) > 1.2)

                            # 4h
                            # (abs(dmp_dmn_diff.iloc[-1]) > 0.5)
                            # and
                            # (abs(globals()['df_4h']['dx_200_diff'].iloc[-1]) > 1.3)




                            # and
                            # ((globals()['df_'+interval]['ADX_200'].iloc[-1]) < 8)


                            # and
                            # ((globals()['df_'+interval]['ADX_200'].iloc[-1]) > 18)



                            # and
                            # ((globals()['df_'+interval]['feature1'].iloc[-1]) > 0)
                            # and
                            # ((globals()['df_'+interval]['ADX_200'].iloc[-1]) > 18)



                            # and
                            # ((globals()['df_'+interval]['feature1'].iloc[-1]) > 0)
                            # and
                            # ((globals()['df_'+interval]['ADX_200'].iloc[-1]) < 8)




                            # # 2.
                            # # ('df_'+interval in globals())
                            # # and
                            # (globals()['df_'+interval]['ADX_200'].count() > 600)
                            # and
                            # # ((globals()['df_1h']['feature1'].iloc[-1]) > 0)
                            # # and
                            # # ((globals()['df_'+interval]['adx_diff'].iloc[-1]) > 0)
                            # # and
                            # # ((globals()['df_'+interval]['dx_diff'].iloc[-1]) > 0)
                            # # and
                            # ((globals()['df_'+interval]['ADX_200'].iloc[-1]) > 10)
                            # # and
                            # # ((globals()['df_'+interval]['DX_14'].iloc[-1]) > (globals()['df_'+interval]['ADX_14'].iloc[-1]))




                            # ((globals()['df_'+interval]['adx_diff'].iloc[-1]) > 0)
                            # and
                            # ((globals()['df_'+interval]['ADX_14'].iloc[-1]) < 18)

                            # ((globals()['df_1h']['feature1'].iloc[-1]) > 0)
                            # and
                            # ((globals()['df_'+interval]['adx_diff'].iloc[-1]) > 0)
                            # and
                            # ((globals()['df_'+interval]['dx_diff'].iloc[-1]) > 0)
                            # and
                            # ((globals()['df_'+interval]['ADX_14'].iloc[-1]) < 18)
                            # and
                            # ((globals()['df_'+interval]['DX_14'].iloc[-1]) > (globals()['df_'+interval]['ADX_14'].iloc[-1]))
                            # and
                            # ((globals()['df_'+interval]['atr_diff'].iloc[-1]) > 0)
                            ###############################################################
                            # ((globals()['df_5m']['atr_diff'].iloc[-1]) > 0) and
                            # ((globals()['df_5m']['adx_diff'].iloc[-1]) > 0) and
                            # ((globals()['df_15m']['atr_diff'].iloc[-1]) > 0) and
                            # ((globals()['df_15m']['adx_diff'].iloc[-1]) > 0) and
                            # ((globals()['df_1h']['atr_diff'].iloc[-1]) > 0) and
                            # ((globals()['df_1h']['adx_diff'].iloc[-1]) > 0)
                            # (
                            #     ((globals()['df_1m']['ATRr_14'].iloc[-1]) > ((globals()['df_1m']['close'].iloc[-1])*0.7/100)) or
                            #     ((globals()['df_5m']['ATRr_14'].iloc[-1]) > ((globals()['df_5m']['close'].iloc[-1])*0.9/100))
                            # )
                        )

                        # print(f'{market}, {condition}, {(globals()["df_1h"]["atr_p_diff"].iloc[-1])}, {(globals()["df_1h"]["adx_AMATe_LR_8_21_2"].iloc[-1])}, {(globals()["df_1m"]["ATRr_14"].iloc[-1])}, {((globals()["df_1m"]["close"].iloc[-1])*0.7/100)}')
                        # if abs(float(percentage_change)) < 20: # daily change over 20%
                        if condition: # and (float(abs(percentage_change)) > 0): # daily change over 20%
                            # if len(top_markets) < 6 or abs(float(percentage_change)) > min(top_markets.values()): # reverse=True
                            ###############################################################
                            if ((len(top_markets) < 4) and (abs(float(percentage_change)) != 0)) or ((abs(float(percentage_change)) != 0) and (abs(float(percentage_change)) > min(top_markets.values()))): # reverse=True
                            # if ((len(top_markets) < 4) and (abs(float(percentage_change)) != 0) and (abs(float(percentage_change)) > 2)) or ((abs(float(percentage_change)) < max(top_markets.values())) and (abs(float(percentage_change)) != 0) and (abs(float(percentage_change)) > 2)): # reverse=False
                            ###############################################################
                            # if (len(top_markets) < 5) or ((abs(float(percentage_change)) != 0) and (abs(float(percentage_change)) < max(top_markets.values()))): # reverse=False
                                top_markets[symbol] = abs(float(percentage_change))
                                ###############################################################
                                top_markets = dict(sorted(top_markets.items(), key=lambda item: item[1], reverse=True)[:4])
                                # top_markets = dict(sorted(top_markets.items(), key=lambda item: item[1], reverse=False)[:4])
                                ###############################################################
                                # top_markets = dict(sorted(top_markets.items(), key=lambda item: item[1], reverse=True)[:3])
                                # top_markets = dict(sorted(top_markets.items(), key=lambda item: item[1], reverse=False)[:5])

                except Exception as e:
                    continue

        # Sort top markets by percentage change in descending order
        sorted_markets = sorted(top_markets.items(), key=lambda item: item[1], reverse=True)
        # Create current_max_changes dictionary with ranks assigned to markets
        current_max_changes = {str(c + 1): market for c, (market, _) in enumerate(sorted_markets)}

        print('top_markets:', top_markets)
        print('top_markets_re:', current_max_changes)

        # 1. out ranked markets
        out_ranked_markets = {rank: market for rank, market in running_markets.items() if market not in current_max_changes.values()}

        # 2. new ranked markets
        new_ranked_markets = {rank: market for rank, market in current_max_changes.items() if market not in running_markets.values()}

        # # 3. markets_to_start
        # for key, value in zip(running_markets, new_ranked_markets.values()):
        #     markets_to_start[key] = value
        available_slots = get_available_slots(base_markets, running_markets, new_ranked_markets, out_ranked_markets)
        markets_to_start = get_markets_to_start(available_slots, new_ranked_markets)

        # 4. Updated running_markets
        for key in out_ranked_markets:
            del running_markets[key]

        running_markets.update(markets_to_start)
        # running_markets = {k: v for k, v in sorted(running_markets.items(), key=lambda item: int(item[0]))}

        print('new_ranked_markets', new_ranked_markets)
        print('--------------')
        print('out_ranked_markets', out_ranked_markets)
        print("markets_to_start:", markets_to_start)
        print('\n--------------')
        print("Updated running_markets:", running_markets)
        print('\n--------------')




        for key, value in out_ranked_markets.items():
            p_id_list_len, p_id_list = ps_counter(key)
            if p_id_list_len > 0:
                for p_id in p_id_list:
                    try:
                        psutil.Process(p_id).kill()
                        print(f"{key}, {value}: has stopped1")
                        # send_to_telegram(f"{key}, {value}: has stopped")
                    except Exception as e:
                        continue




            if value:
                position_side, position_size, position_value, position_entry_price, liquidation_price, unrealised_pnl, roe_pcnt = position_calc(exchange_id, value)
                if position_size: # 포지션존재 1.

                    # # if unrealised_pnl < 0: # 손실 중일 때 포지션 종료
                    # if ((unrealised_pnl / wallet_balance) * 100 >= 0.13): # 수익 중일 때 포지션 종료
                    #     exchange.cancel_all_orders(value)
                    #     print(f"{key}, {value}: all open orders canceled")
                    #     if position_side == 'long':
                    #         exit_side = 'sell'
                    #         limit_type_only_case_price_pick_param = float(1 - 0.0005)
                    #     elif position_side == 'short':
                    #         exit_side = 'buy'
                    #         limit_type_only_case_price_pick_param = float(1 + 0.0005)
                    #     exchange.create_order(symbol=value, type='market', side=exit_side, amount=position_size, params={'reduceOnly': True}) # 손절
                    #     print(f"{key}, {value}: Engin 종료로 인한, position closed")
                    #     send_to_telegram(f"{key}, {value}: Engin 종료로 인한, position closed")

                    # # if (unrealised_pnl / wallet_balance) * 100 <= -1: # 손실이 지갑의 3% 초과 시 반대진입
                    # #     exchange.cancel_all_orders(value)
                    # #     print(f"{key}, {value}: all open orders canceled")
                    # #     if position_side == 'long':
                    # #         exit_side = 'sell'
                    # #         limit_type_only_case_price_pick_param = float(1 - 0.0005)
                    # #     elif position_side == 'short':
                    # #         exit_side = 'buy'
                    # #         limit_type_only_case_price_pick_param = float(1 + 0.0005)
                    # #     exchange.create_order(symbol=value, type='market', side=exit_side, amount=position_size*2, params=params) # 손절
                    # #     print(f"{key}, {value}: 손실이 지갑의 1% 초과 시 반대진입")
                    # #     send_to_telegram(f"{key}, {value}: 손실이 지갑의 3% 초과 시 반대진입")

                    # # else: # 수익 중일 때 포지션 유지
                    # # exchange.cancel_all_orders(value)
                    # # print(f"{key}, {value}: all open orders canceled")
                    # # if position_side == 'long':
                    # #     exit_side = 'sell'
                    # #     limit_type_only_case_price_pick_param = float(1 - 0.0005)
                    # # elif position_side == 'short':
                    # #     exit_side = 'buy'
                    # #     limit_type_only_case_price_pick_param = float(1 + 0.0005)
                    # # exchange.create_order(symbol=value, type='market', side=exit_side, amount=position_size*2, params=params) # 손절
                    # # print(f"{key}, {value}: 손실이 지갑의 3% 초과 시 반대진입")
                    # # send_to_telegram(f"{key}, {value}: 손실이 지갑의 3% 초과 시 반대진입")


                    print(f"{key}, {value}: position kept")
                    send_to_telegram(f"{key}, {value}: position kept")











                # exchange.cancel_all_orders(value)
                # if position_size: # 포지션존재 1.
                #     if position_side == 'long':
                #         exit_side = 'sell'
                #         limit_type_only_case_price_pick_param = float(1 - 0.0005)
                #     elif position_side == 'short':
                #         exit_side = 'buy'
                #         limit_type_only_case_price_pick_param = float(1 + 0.0005)
                #     exchange.create_order(symbol=value, type='market', side=exit_side, amount=position_size, params={'reduceOnly': True}) # 손절
                #     print(f"{key}, {value}: position closed")
                #     send_to_telegram(f"{key}, {value}: position closed")
            # if value:
                
            #     position_side, position_size, position_value, position_entry_price, liquidation_price, unrealised_pnl, roe_pcnt = position_calc(exchange_id, value)
            #     if position_size: # 포지션존재 1.
            #         if unrealised_pnl >= 0: # 수익 중일 때 포지션 종료
            #             exchange.cancel_all_orders(value)
            #             print(f"{key}, {value}: all open orders canceled")
            #             # send_to_telegram(f"{key}, {value}: all open orders canceled")

            #             if position_side == 'long':
            #                 exit_side = 'sell'
            #             elif position_side == 'short':
            #                 exit_side = 'buy'
            #             exchange.create_order(symbol=value, type='market', side=exit_side, amount=position_size, params={'reduceOnly': True}) # 손절
            #             print(f"{key}, {value}: position closed")
            #             send_to_telegram(f"{key}, {value}: position closed")
            #         else: # 손실 중일 때 반대진입 후 트레이드 종료 
            #             # print(f"{key}, {value}: position kept")
            #             # send_to_telegram(f"{key}, {value}: position kept")

            #             exchange.cancel_all_orders(value)
            #             print(f"{key}, {value}: all open orders canceled")
            #             # send_to_telegram(f"{key}, {value}: all open orders canceled")

            #             if position_side == 'long':
            #                 exit_side = 'sell'
            #             elif position_side == 'short':
            #                 exit_side = 'buy'

            #             exit_order_position_amount_re = (position_size * 2)

            #             exchange.create_order(symbol=value, type='market', side=exit_side, amount=exit_order_position_amount_re, params=params) # 반대진입
            #             print(f"{key}, {value}: position 반대진입 후 매매 종료")
            #             send_to_telegram(f"{key}, {value}: position 반대진입 후 매매 종료")










        # open_positions = exchange.fetch_positions()
        # open_positions_count = len(open_positions)

        # for position in open_positions:
        #     unrealised_pnl_ = position['unrealizedPnl']
        #     position_side_ = position['side']
        #     symbol_ = position['symbol']
        #     position_size_ = position['contracts']
        #     position_entry_price_ = position["entryPrice"]
        #     position_entry_time_ = position["timestamp"]/1000
            



        #     if symbol_ not in markets_to_start and symbol_ not in running_markets:
        #         # 손실이 지갑의 3% 초과 시 반대진입
        #         # if (unrealised_pnl_ / wallet_balance) * 100 <= -5:  # 손실이 3% 초과 시
        #         #     exchange.cancel_all_orders(symbol_)
        #         #     print(f"{symbol_}: all open orders canceled")
                    
        #         #     if position_side_ == 'long':
        #         #         exit_side = 'sell'
        #         #         limit_type_only_case_price_pick_param = float(1 - 0.0005)
        #         #     elif position_side_ == 'short':
        #         #         exit_side = 'buy'
        #         #         limit_type_only_case_price_pick_param = float(1 + 0.0005)

        #         #     print(f"{symbol_}: 손실이 지갑의 -0.5% 초과 하여 반대진입")
        #         #     print(exit_side)
        #         #     print(position_size_*2)
        #         #     # 반대진입, 포지션 크기의 2배로 진입
        #         #     exchange.create_order(symbol=symbol_, type='market', side=exit_side, amount=position_size_*2, params={})
                    
        #         #     send_to_telegram(f"{symbol_}: 손실이 지갑의 -0.5% 초과 하여 반대진입")
        




        #         if ((unrealised_pnl_ / wallet_balance) * 100 >= 2) or ((unrealised_pnl_ / wallet_balance) * 100 < -2):  # 이익이 3% 이상이거나 손실이 3% 이하인 경우

        #             exchange.cancel_all_orders(symbol_)
        #             print(f"{symbol_}: all open orders canceled")
                    
        #             if position_side_ == 'long':
        #                 exit_side = 'sell'
        #                 limit_type_only_case_price_pick_param = float(1 - 0.0005)
        #             elif position_side_ == 'short':
        #                 exit_side = 'buy'
        #                 limit_type_only_case_price_pick_param = float(1 + 0.0005)

        #             print(f"{symbol_}: 이익이 지갑의 2% 초과, 익절")
        #             print(exit_side)
        #             print(position_size_)
        #             # 반대진입, 포지션 크기의 2배로 진입
        #             exchange.create_order(symbol=symbol_, type='market', side=exit_side, amount=position_size_, params={'reduceOnly': True})
                    
        #             send_to_telegram(f"{symbol_}: 이익이 지갑의 2% 초과, 익절")













        #             # exchange.cancel_all_orders(symbol_)

        #             # symbol_ticker_last = ticker_calc(symbol_)[1]
        #             # peak_calc(symbol_, intervals_2, 1000)
        #             # cumulate_lv = cumulate_lv_calc(symbol_, intervals_2)
        #             # pick_max, pick_min, previous_price_frequency, greedy_percentage, picked_interval = pick_calc(cumulate_lv, position_side_, position_size_, position_entry_price_, symbol_ticker_last, '')
        #             # l_p_l, l_p_h, last_peaked_price = l_p_l_and_l_p_h_calc(symbol_ticker_last)
        #             # stric_exit_min_price = stric_exit_price_calc_min(position_side_, position_entry_price_)


        #             # stop_loss_, percentage_difference_stop_loss_, target_p_, percentage_difference_target_p_ = stop_loss_calc('stg_type', 'success', position_side_, position_entry_price_, symbol_ticker_last, 'close_price_low', 'close_price_high', pick_max, pick_min, 'stop_loss_range')
                    
        #             # if position_side_ == 'long': # 포지션 롱 일때
        #             #     if stop_loss_ < symbol_ticker_last: # 현재 포지가 long 일때
        #             #         exit_order_position_amount = exit_order_position_amount_calc(position_size_)
        #             #         exchange.create_order(symbol=symbol_, type='stop_market', side='sell', amount=exit_order_position_amount, params={'stopPrice': stop_loss_}) # stop market 재생성
        #             #         exit_status = '1-4-2. 역지'
        #             #         message = f'이익이 3% 초과 하여 익절 역지 생성 하였습니다.'
        #             #         message += ' [2:1]: ' + str((pick_max - position_entry_price_)/stopPrice_const) + ', 손절가: ' + str(position_entry_price_ - ((pick_max - position_entry_price_)/stopPrice_const))
        #             #         # check_pointer(trader_name, exchange_id, symbol_, balance_currency, loop, max_waiting_in_second, exit_status, message)
        #             #         print(message)
        #             #         send_to_telegram(f"{symbol_}: 이익이 3% 초과 하여 익절 역지 생성")


        #             # elif position_side_ == 'short': # 포지션 롱 일때
        #             #     if stop_loss_ > symbol_ticker_last: # 현재 포지가 short 일때
        #             #         exit_order_position_amount = exit_order_position_amount_calc(position_size_)
        #             #         exchange.create_order(symbol=symbol_, type='stop_market', side='buy', amount=exit_order_position_amount, params={'stopPrice': stop_loss_}) # stop market 재생성
        #             #         exit_status = '1-4-8. 역지'
        #             #         message = f'이익이 3% 초과 하여 익절 역지 생성 하였습니다.'
        #             #         message += '[2:1]: ' + str((position_entry_price_ - pick_min)/stopPrice_const) + ', 손절가: ' + str(position_entry_price_ + ((position_entry_price_ - pick_min)/stopPrice_const))
        #             #         # check_pointer(trader_name, exchange_id, market_id, balance_currency, loop, max_waiting_in_second, exit_status, message)
        #             #         print(message)
        #             #         send_to_telegram(f"{symbol_}: 이익이 3% 초과 하여 익절 역지 생성")




















        #         # elif (unrealised_pnl_ / wallet_balance) * 100 < -3:  # 이익이 3% 초과 시

        #         #     # exchange.cancel_all_orders(symbol_)

        #         #     # position_side, position_size, position_value, position_entry_price, liquidation_price, unrealised_pnl, roe_pcnt = position_calc(exchange_id, value)
        #         #     # if position_size: # 포지션존재 1.

        #         #     #     # if unrealised_pnl < 0: # 손실 중일 때 포지션 종료
        #         #     #     exchange.cancel_all_orders(value)
        #         #     #     print(f"{key}, {value}: all open orders canceled")
        #         #     #     if position_side == 'long':
        #         #     #         exit_side = 'sell'
        #         #     #         limit_type_only_case_price_pick_param = float(1 - 0.0005)
        #         #     #     elif position_side == 'short':
        #         #     #         exit_side = 'buy'
        #         #     #         limit_type_only_case_price_pick_param = float(1 + 0.0005)
        #         #     #     exchange.create_order(symbol=value, type='market', side=exit_side, amount=position_size, params={'reduceOnly': True}) # 손절
        #         #     #     print(f"{key}, {value}: position closed")
        #         #     #     send_to_telegram(f"{key}, {value}: position closed")


        #         # # 손실이 지갑의 3% 초과 시 반대진입
        #         # # if (unrealised_pnl_ / wallet_balance) * 100 <= -5:  # 손실이 3% 초과 시




                
        #         #     exchange.cancel_all_orders(symbol_)
        #         #     print(f"{symbol_}: all open orders canceled")
                    
        #         #     if position_side_ == 'long':
        #         #         exit_side = 'sell'
        #         #         limit_type_only_case_price_pick_param = float(1 - 0.0005)
        #         #     elif position_side_ == 'short':
        #         #         exit_side = 'buy'
        #         #         limit_type_only_case_price_pick_param = float(1 + 0.0005)

        #         #     print(f"{symbol_}: 손실이 지갑의 -3% 초과 하여 반대진입")
        #         #     print(exit_side)
        #         #     print(position_size_*2)
        #         #     # 반대진입, 포지션 크기의 2배로 진입
        #         #     exchange.create_order(symbol=symbol_, type='market', side=exit_side, amount=position_size_*2, params={})
                    
        #         #     send_to_telegram(f"{symbol_}: 손실이 지갑의 -3% 초과 하여 반대진입")
        





        for key, value in markets_to_start.items():
            cmd = "nohup python3 " + key + " " + value + " {0} >/dev/null 2>&1 &"
            subprocess.Popen(cmd, shell=True)
            print(f"{key}, {value}: has started")
            send_to_telegram(f"{key}, {value}: has started")
    else:
        print(f"currently open position is over {open_positions_count}, no more new engin trying!")
        send_to_telegram(f"currently open position is over {open_positions_count}, no more new engin trying!")



    end = time.time()
    total_time = end - start
    print(str(total_time/60) + ' min')
    
    formatted_datetime = datetime.datetime.now().replace(microsecond=0)
    print(formatted_datetime)
    print('\n')

    if len(top_markets) < 3:
        time.sleep(60)
    else:
        time.sleep(25000)
